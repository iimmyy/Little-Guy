#!/usr/bin/env python3
"""Votes of every write at all decisive digit spots, right token vs his 3 best wrong. Reading only.
usage: sum_read.py <hf> <steps.jsonl> <out prefix> [limit]"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, steps, prefix = sys.argv[1:4]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
DEV, MAXTOK = "cuda", 16000
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


V3, VS, Vg, rows, grows = [], [], [], [], []


def votes(W, r, x, y):
    return ((W @ (c * (U[x] - U[y]))) / r).cpu().numpy().astype(np.float32)


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
    rng = np.random.default_rng(int(j["step"]))
    low = max(5, first - 1500)
    gen = rng.choice(np.arange(low, first - 1), size=2, replace=False).tolist() if first - 1 - low >= 2 else []
    logits, W, r = forward(ids, [q - 1 for q, _, _ in cands] + gen)
    for n, (q, pos, s) in enumerate(cands):
        l, pool = logits[n], LEN[len(s)]
        bs = []
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
                bs.append(i)
                if len(bs) == 3:
                    break
        if not bs:
            continue
        a = ids[q]
        arr = np.full((3, 49), np.nan, dtype=np.float32)
        for z, b in enumerate(bs):
            arr[z] = votes(W[n], r[n], a, b)
        top = l.topk(2).indices.tolist()
        V3.append(arr)
        VS.append(votes(W[n], r[n], top[0], top[1]))
        lp = l.log_softmax(0)
        rows.append(dict(step=j["step"], id=j["id"], run=j["run"], split=j["split"], seen=j["seen_before"], q=q, pos=pos, a=tok.decode([a]), b=[tok.decode([b]) for b in bs], m=[float(l[a] - l[b]) for b in bs], pa=float(lp[a].exp()), top=tok.decode([top[0]]), m_self=float(l[top[0]] - l[top[1]])))
    for z, p in enumerate(gen):
        n = len(cands) + z
        top = logits[n].topk(2).indices.tolist()
        Vg.append(votes(W[n], r[n], top[0], top[1]))
        grows.append(dict(step=j["step"], id=j["id"], split=j["split"], p=int(p), m_self=float(logits[n][top[0]] - logits[n][top[1]])))
    return True


J = [json.loads(l) for l in open(steps)]
if LIMIT:
    J = J[:LIMIT]
t0 = time.time()
skipped = 0
for n, j in enumerate(J):
    ok = right_step(j)
    skipped += not ok
    if n % 250 == 0:
        print(n, len(J), "spots", len(rows), "general", len(grows), "skipped", skipped, round(time.time() - t0), flush=True)
np.savez(prefix + "-x.npz", V3=np.stack(V3), VS=np.stack(VS), Vg=np.stack(Vg), order=np.array(order))
with open(prefix + "-rows.jsonl", "w") as w:
    for r_ in rows:
        w.write(json.dumps(r_) + "\n")
with open(prefix + "-general.jsonl", "w") as w:
    for r_ in grows:
        w.write(json.dumps(r_) + "\n")
print("SUM_DONE steps", len(J) - skipped, "skipped", skipped, "spots", len(rows), "general", len(grows), round(time.time() - t0), flush=True)
