#!/usr/bin/env python3
"""Is the page's pull a straight line, and is K what he knows? Reading only. usage: line_read.py <hf> <steps.jsonl> <out.jsonl> <split> <limit>"""
import json, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, steps, outf, split, limit = sys.argv[1:6]
limit = int(limit)
DEV, MAXTOK = "cuda", 7000
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
EMB = m.model.embed_tokens
EXPR = re.compile(r"(?<![\d.,])\d[\d.,]*\s*(?:\*|×|\\times|\\cdot|/|\+|-)\s*\d[\d.,]*\s*=\s*$")
WAIT, YES = [tok(x, add_special_tokens=False).input_ids[0] for x in (" Wait", " Yes")]
print("mark tokens", repr(tok.decode([WAIT])), repr(tok.decode([YES])), flush=True)


def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"


@torch.inference_mode()
def read(E, a, b):
    l = m(inputs_embeds=E[None], logits_to_keep=1).logits[0, -1].float()
    return float(l[a] - l[b])


J = [json.loads(l) for l in open(steps)]
J = [j for j in J if j["split"] == split and "broken" in j and EXPR.search(j["pre"])][:limit]
t0, n = time.time(), 0
with open(outf, "w") as w:
    for j in J:
        h = head(j["q"])
        written, right = j["broken"], j["clean"]
        expr = EXPR.search(j["pre"]).group()
        W = tok(h + j["pre"] + written, add_special_tokens=False).input_ids
        R = tok(h + j["pre"] + right, add_special_tokens=False).input_ids
        diff = [i for i, (x, y) in enumerate(zip(W, R)) if x != y]
        if len(W) != len(R) or len(diff) != 1 or len(W) > MAXTOK:
            continue
        q = diff[0]
        a, b = R[q], W[q]
        L0 = len(tok(h + j["pre"], add_special_tokens=False).input_ids)
        shared = tok.decode(R[L0:q]) if q > L0 else ""
        tail = ". " + expr + shared
        idsW = tok(h + j["pre"] + written + tail, add_special_tokens=False).input_ids
        idsR = tok(h + j["pre"] + right + tail, add_special_tokens=False).input_ids
        if idsW[:len(W)] != W or idsR[:len(R)] != R:
            continue
        EW, ER = EMB(torch.tensor(idsW, device=DEV)), EMB(torch.tensor(idsR, device=DEV))
        va, vw, vy = [EMB(torch.tensor(x, device=DEV)) for x in (a, WAIT, YES)]
        reads = {}
        reads["plain:W"], reads["plain:R"] = read(EW, a, b), read(ER, a, b)
        for phi in (0.25, 0.75):
            E = EW.clone()
            E[q] = (1 - phi) * E[q] + phi * va
            reads[f"phi{phi}:W"] = read(E, a, b)
        for name, v in (("wait", vw), ("yes", vy)):
            for page, E0 in (("W", EW), ("R", ER)):
                E = E0.clone()
                E[q] = 0.75 * E[q] + 0.25 * v
                reads[f"mark{name}:{page}"] = read(E, a, b)
        reads["m0"] = read(EMB(torch.tensor(R[:q], device=DEV)), a, b)
        w.write(json.dumps(dict(step=j["step"], id=j["id"], split=j["split"], written=written, right=right, reads=reads)) + "\n")
        w.flush()
        n += 1
        if n % 20 == 0:
            print(n, len(J), round(time.time() - t0), flush=True)
print("LINE_DONE", n, round(time.time() - t0), flush=True)
