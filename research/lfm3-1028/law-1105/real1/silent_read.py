#!/usr/bin/env python3
"""A hidden second look at a number he wrote. Reading only; nothing is added to his text, no weights changed.
usage: silent_read.py <hf> <steps.jsonl> <out.jsonl>"""
import json, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, steps, outf = sys.argv[1:4]
DEV, MAXTOK, NEW = "cuda", 9000, 14
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
EOS = {tok.eos_token_id} | set(m.generation_config.eos_token_id if isinstance(m.generation_config.eos_token_id, list) else [m.generation_config.eos_token_id])
NUM = re.compile(r"\s*(-?\d[\d,]*(?:\.\d+)?)")
SEPS = {"plain": ". ", "now": ". Now, ", "wait": ". Wait, "}


def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"


@torch.inference_mode()
def go(text):
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) > MAXTOK:
        return None, None
    o = m(input_ids=torch.tensor([ids], device=DEV), use_cache=True, logits_to_keep=1)
    pkv, out = o.past_key_values, []
    t = int(o.logits[0, -1].argmax())
    for _ in range(NEW):
        if t in EOS:
            break
        out.append(t)
        o = m(input_ids=torch.tensor([[t]], device=DEV), past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv = o.past_key_values
        t = int(o.logits[0, -1].argmax())
    s = tok.decode(out)
    n = NUM.match(s)
    return s, (n.group(1).replace(",", "") if n else None)


J = [json.loads(l) for l in open(steps)]
t0 = time.time()
with open(outf, "w") as w:
    for i, j in enumerate(J):
        h = head(j["q"])
        row = {k: j[k] for k in ("step", "id", "run", "kind", "written", "true", "qmark", "doubt_next", "rel")}
        row["expr"] = j["expr"]
        row["reads"] = {}
        s, n = go(h + j["pre"])
        row["reads"]["first"] = (n, s)
        for name, sep in SEPS.items():
            s, n = go(h + j["pre"] + j["written"] + sep + j["expr"])
            row["reads"][name] = (n, s)
        w.write(json.dumps(row) + "\n")
        w.flush()
        if i % 20 == 0:
            print(i, len(J), round(time.time() - t0), flush=True)
print("SILENT_DONE", len(J), round(time.time() - t0), flush=True)
