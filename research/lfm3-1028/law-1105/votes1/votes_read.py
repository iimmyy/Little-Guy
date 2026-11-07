#!/usr/bin/env python3
"""Exact vote of every write at digit tokens. Reading only.
usage: votes_read.py <hf> <steps.jsonl> <out prefix> [limit]"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, steps, prefix = sys.argv[1:4]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
DEV, MAXTOK, BAND = "cuda", 16000, 6.0
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
LS = m.model.layers
N = len(LS)
U = m.lm_head.weight.float()
c = m.model.embedding_norm.weight.float()
eps = m.model.embedding_norm.variance_epsilon
assert m.lm_head.weight is m.model.embed_tokens.weight or torch.equal(m.lm_head.weight, m.model.embed_tokens.weight)
IDX, got = {}, {}


def save(name):
    def f(mod, a, o=None):
        v = a[0] if o is None else o
        got[name] = v[0][IDX["i"]].float()
    return f


LS[0].operator_norm.register_forward_pre_hook(save("x0"))
m.model.embedding_norm.register_forward_pre_hook(save("xN"))
for L, b in enumerate(LS):
    b.post_operator_norm.register_forward_hook(save(f"op{L}"))
    b.post_ffn_norm.register_forward_hook(save(f"ffn{L}"))
order = ["x0"] + [n_ for L in range(N) for n_ in (f"op{L}", f"ffn{L}")]
LEN = {}
for i in range(U.shape[0]):
    s = tok.decode([i])
    if s.isdigit() and s.isascii():
        LEN.setdefault(len(s), []).append(i)
LEN = {k: torch.tensor(v, device=DEV) for k, v in LEN.items()}
print("digit tokens by length", {k: len(v) for k, v in LEN.items()}, flush=True)


def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"


def forward(ids, qs):
    IDX["i"] = torch.tensor(qs, device=DEV)
    with torch.inference_mode():
        hs = m.model(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0][IDX["i"]].float()
        logits = hs @ U.T
        W = torch.stack([got[n_] for n_ in order], 1)
        r = torch.sqrt(got["xN"].pow(2).mean(-1, keepdim=True) + eps)
    return logits, W, r


V_, S_, R_, rows, all_m = [], [], [], [], []


def record(j, l, W, r, a, b, q, pos, kind):
    mm = float(l[a] - l[b])
    lp = l.log_softmax(0)
    d = c * (U[a] - U[b])
    V = (W @ d) / r
    V_.append(V.cpu().numpy().astype(np.float32))
    S_.append(W.norm(dim=1).cpu().numpy().astype(np.float32))
    R_.append(W.cumsum(0).pow(2).mean(1).sqrt().cpu().numpy().astype(np.float32))
    rows.append(dict(step=j["step"], id=j["id"], run=j["run"], split=j["split"], seen=j["seen_before"], kind=kind, q=q, pos=pos, a=tok.decode([a]), b=tok.decode([b]),
                     m=mm, tally=float(V.sum()), pa=float(lp[a].exp()), pb=float(lp[b].exp()), bits=float(-(lp.exp() * lp).sum() / np.log(2)), dsize=float(d.norm()), r=float(r)))


def wrong(v, t, ulp):
    e = abs(v - t)
    return t != 0 and e / abs(t) >= 0.02 and e > 2 * ulp


def right_step(j):
    h = head(j["q"])
    st, d, t = len(h) + len(j["pre"]), j["number"], j["true"]
    enc = tok(h + j["pre"] + d, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc.input_ids, enc.offset_mapping
    first = next((i for i, (x, y) in enumerate(off) if y > st), None)
    if first is None or first == 0 or off[first][0] != st or len(ids) > MAXTOK:
        return False
    clean = d.replace(",", "")
    ulp = 10 ** (-len(clean.split(".")[1])) if "." in clean else 1
    cands = []
    for q in range(first, len(ids)):
        s = tok.decode([ids[q]])
        if s.isdigit() and s.isascii() and s == d[off[q][0] - st:off[q][1] - st] and len(s) in LEN:
            cands.append((q, off[q][0] - st, s))
    if not cands:
        return True
    logits, W, r = forward(ids, [q - 1 for q, _, _ in cands])
    for n, (q, pos, s) in enumerate(cands):
        l, pool = logits[n], LEN[len(s)]
        b = None
        for i in pool[l[pool].topk(min(30, len(pool))).indices].tolist():
            if i == ids[q]:
                continue
            alt = tok.decode([i])
            if pos == (1 if d.startswith("-") else 0) and alt[0] == "0" and "." not in d[:pos + len(alt) + 1]:
                continue
            try:
                v = float((d[:pos] + alt + d[pos + len(alt):]).replace(",", ""))
            except ValueError:
                continue
            if wrong(v, t, ulp):
                b = i
                break
        if b is None:
            continue
        mm = float(l[ids[q]] - l[b])
        all_m.append(mm)
        if abs(mm) <= BAND:
            record(j, l, W[n], r[n], ids[q], b, q, pos, "right")
    return True


def slip_step(j):
    h = head(j["q"])
    S = tok(h + j["pre"] + j["slip"], add_special_tokens=False).input_ids
    F = tok(h + j["pre"] + j["fixed"], add_special_tokens=False).input_ids
    q = next(i for i, (x, y) in enumerate(zip(S, F)) if x != y)
    logits, W, r = forward(S[:q], [q - 1])
    record(j, logits[0], W[0], r[0], F[q], S[q], q, -1, "slip")
    return True


J = [json.loads(l) for l in open(steps)]
if LIMIT:
    J = J[:LIMIT]
t0 = time.time()
skipped = 0
for n, j in enumerate(J):
    ok = slip_step(j) if "slip" in j else right_step(j)
    skipped += not ok
    if n % 250 == 0:
        print(n, len(J), "spots", len(rows), "decisive", len(all_m), "skipped", skipped, round(time.time() - t0), flush=True)
np.savez(prefix + "-x.npz", V=np.stack(V_), S=np.stack(S_), R=np.stack(R_), order=np.array(order), all_m=np.array(all_m, dtype=np.float32))
with open(prefix + "-rows.jsonl", "w") as w:
    for r_ in rows:
        w.write(json.dumps(r_) + "\n")
e = np.array([abs(r_["m"] - r_["tally"]) for r_ in rows])
print("VOTES_DONE steps", len(J) - skipped, "skipped", skipped, "spots in band", len(rows), "decisive", len(all_m), "tally minus real gap: median", round(float(np.median(e)), 4), "worst", round(float(e.max()), 4), round(time.time() - t0), flush=True)
