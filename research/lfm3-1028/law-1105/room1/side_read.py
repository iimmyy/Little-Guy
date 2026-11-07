#!/usr/bin/env python3
"""He continues after a wrong number, with or without a side word at that token. No weights changed.
usage: side_read.py <hf> <steps.jsonl> <out.jsonl> <split or all> <limit> <phis comma> [new tokens]"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, steps, outf, split, limit, phis = sys.argv[1:7]
limit = int(limit)
phis = [float(x) for x in phis.split(",")]
NEW = int(sys.argv[7]) if len(sys.argv) > 7 else 100
DEV = "cuda"
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
EMB = m.model.embed_tokens
EOS = {tok.eos_token_id} | set(m.generation_config.eos_token_id if isinstance(m.generation_config.eos_token_id, list) else [m.generation_config.eos_token_id])
LEN = {}
for i in range(m.lm_head.weight.shape[0]):
    s = tok.decode([i])
    if s.isdigit() and s.isascii():
        LEN.setdefault(len(s), []).append(i)


def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"


@torch.inference_mode()
def go(E):
    o = m(inputs_embeds=E[None], use_cache=True, logits_to_keep=1)
    pkv, out = o.past_key_values, []
    t = int(o.logits[0, -1].argmax())
    for _ in range(NEW):
        if t in EOS:
            break
        out.append(t)
        o = m(input_ids=torch.tensor([[t]], device=DEV), past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv = o.past_key_values
        t = int(o.logits[0, -1].argmax())
    return out


@torch.inference_mode()
def cache_check(ids):
    """Cached steps must match a full pass."""
    E = EMB(torch.tensor(ids, device=DEV))
    out = go(E)[:6]
    full = list(ids)
    for t in out:
        o = m(input_ids=torch.tensor([full], device=DEV), logits_to_keep=1)
        if int(o.logits[0, -1].argmax()) != t:
            return False
        full.append(t)
    return True


J = [json.loads(l) for l in open(steps)]
if split != "all":
    J = [j for j in J if j["split"] == split]
t0 = time.time()
done = skipped = toks = 0
checked = False
with open(outf, "w") as w:
    for j in J:
        if done >= limit:
            break
        written, right = (j["slip"], j["fixed"]) if "slip" in j else (j["broken"], j["clean"])
        h = head(j["q"])
        W = tok(h + j["pre"] + written, add_special_tokens=False).input_ids
        R = tok(h + j["pre"] + right, add_special_tokens=False).input_ids
        diff = [i for i, (x, y) in enumerate(zip(W, R)) if x != y]
        if "slip" in j:
            if not diff:
                skipped += 1
                continue
            q = diff[0]
            a, b = R[q], W[q]
        else:
            if len(W) != len(R) or len(diff) != 1 or len(W) > 7000:
                skipped += 1
                continue
            q = diff[0]
            a, b = R[q], W[q]
        if not checked:
            print("cache matches a full pass:", cache_check(W), flush=True)
            checked = True
        rng = np.random.default_rng(int(j["step"]))
        sa = tok.decode([a])
        pool = [i for i in LEN.get(len(sa), LEN[1]) if i not in (a, b) and (sa[0] == "0" or tok.decode([i])[0] != "0")]
        o = int(rng.choice(pool))
        ti = torch.tensor(W, device=DEV)
        base = EMB(ti).clone()
        ua, ub, uo = EMB(torch.tensor(a, device=DEV)), EMB(torch.tensor(b, device=DEV)), EMB(torch.tensor(o, device=DEV))
        arms = [("A", 0.0, None)] + [(n, p, u) for p in phis for n, u in (("B", ua), ("C", uo))]
        for name, p, u in arms:
            E = base.clone()
            if u is not None:
                E[q] = (1 - p) * ub + p * u
            out = go(E)
            toks += len(out)
            w.write(json.dumps(dict(step=j["step"], id=j["id"], split=j["split"], arm=name, phi=p, written=written, right=right, a=sa, b=tok.decode([b]), o=tok.decode([o]), new=len(out), text=tok.decode(out))) + "\n")
        w.flush()
        done += 1
        if done % 5 == 0:
            print(done, "skipped", skipped, "tokens", toks, "per second", round(toks / (time.time() - t0), 1), round(time.time() - t0), flush=True)
print("SIDE_DONE steps", done, "skipped", skipped, "tokens", toks, round(time.time() - t0), flush=True)
