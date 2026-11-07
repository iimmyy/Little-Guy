#!/usr/bin/env python3
"""Does a word with nothing in it move his belief? Reading only, nothing changed in the model.
usage: empty_read.py <hf> <steps.jsonl> <out.jsonl> <split> <limit>"""
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
ARMS = {"plain": ". ", "so": ". So ", "now": ". Now, ", "wait": ". Wait, ", "hmm": ". Hmm, ", "actually": ". Actually, ", "yes": ". Yes, ", "right": ". That's right. ", "wrong": ". That's wrong. "}


def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"


@torch.inference_mode()
def read(E, a, b):
    l = m(inputs_embeds=E[None], logits_to_keep=1).logits[0, -1].float()
    lp = l.log_softmax(0)
    return float(l[a] - l[b]), float(lp[a].exp()), float(lp[b].exp())


J = [json.loads(l) for l in open(steps)]
J = [j for j in J if j["split"] == split and "broken" in j and EXPR.search(j["pre"])]
print("usable steps", len(J), flush=True)
J = J[:limit]
t0, skipped, n = time.time(), 0, 0
with open(outf, "w") as h_:
    for j in J:
        h = head(j["q"])
        written, right = j["broken"], j["clean"]
        expr = EXPR.search(j["pre"]).group()
        W = tok(h + j["pre"] + written, add_special_tokens=False).input_ids
        R = tok(h + j["pre"] + right, add_special_tokens=False).input_ids
        diff = [i for i, (x, y) in enumerate(zip(W, R)) if x != y]
        if len(W) != len(R) or len(diff) != 1 or len(W) > MAXTOK:
            skipped += 1
            continue
        q = diff[0]
        a, b = R[q], W[q]
        L0 = len(tok(h + j["pre"], add_special_tokens=False).input_ids)
        shared = tok.decode(R[L0:q]) if q > L0 else ""
        row = dict(step=j["step"], id=j["id"], split=j["split"], written=written, right=right, a=tok.decode([a]), b=tok.decode([b]), expr=expr, reads={})
        ok = True
        for name, sep in ARMS.items():
            for page, num, ids0 in (("R", right, R), ("W", written, W)):
                ids = tok(h + j["pre"] + num + sep + expr + shared, add_special_tokens=False).input_ids
                if ids[:len(ids0)] != ids0:
                    ok = False
                    break
                row["reads"][f"{name}:{page}"] = read(EMB(torch.tensor(ids, device=DEV)), a, b)
            if not ok:
                break
        if not ok:
            skipped += 1
            continue
        ids = tok(h + j["pre"] + written + ARMS["plain"] + expr + shared, add_special_tokens=False).input_ids
        E = EMB(torch.tensor(ids, device=DEV)).clone()
        E[q] = 0.5 * E[q] + 0.5 * EMB(torch.tensor(a, device=DEV))
        row["reads"]["side:W"] = read(E, a, b)
        h_.write(json.dumps(row) + "\n")
        h_.flush()
        n += 1
        if n % 10 == 0:
            print(n, len(J), "skipped", skipped, round(time.time() - t0), flush=True)
print("EMPTY_DONE steps", n, "skipped", skipped, round(time.time() - t0), flush=True)
