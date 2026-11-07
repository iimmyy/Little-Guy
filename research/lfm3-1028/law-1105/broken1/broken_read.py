#!/usr/bin/env python3
"""Reads what every write does right after a broken number. Reading only.
usage: broken_read.py <hf> <steps.jsonl> <out prefix> [limit]"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

hf, steps, prefix = sys.argv[1:4]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
DEV = "cuda"
ARMS = ("clean", "broken", "reformat")
KS = (-1, 0, 1, 2, 3, 5, 10, 20, 50)
MAXTOK = 4000
tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
LS = m.model.layers
N = len(LS)
IDX = {"i": None}
got = {}
W13 = [(b.feed_forward.w1.weight.float(), b.feed_forward.w3.weight.float()) for b in LS]


def save(name):
    def f(mod, a, o=None):
        v = a[0] if o is None else o
        got[name] = v[0][IDX["i"]].float()
    return f


LS[0].operator_norm.register_forward_pre_hook(save("x0"))
LS[-1].ffn_norm.register_forward_pre_hook(save("xlast"))
for L, b in enumerate(LS):
    b.post_operator_norm.register_forward_hook(save(f"op{L}"))
    b.post_ffn_norm.register_forward_hook(save(f"ffn{L}"))
    b.feed_forward.register_forward_pre_hook(save(f"xt{L}"))
order = ["x0"] + [n for L in range(N) for n in (f"op{L}", f"ffn{L}")]
iu = np.triu_indices(len(order))


def prepare(j, arm):
    """Token positions for one arm; None if the number does not fall on token edges."""
    h = tok.apply_chat_template([{"role": "user", "content": j["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
    num = j[arm]
    text = h + j["pre"] + num + j["post"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    st = len(h) + len(j["pre"])
    en = st + len(num)
    off = enc.offset_mapping
    first = next((i for i, (a, b) in enumerate(off) if b > st), None)
    after = next((i for i, (a, b) in enumerate(off) if a >= en), None)
    if first is None or after is None or off[first][0] != st or off[after][0] != en or len(enc.input_ids) > MAXTOK or first == 0:
        return None
    return dict(ids=enc.input_ids, first=first, after=after)


def main():
    J = [json.loads(l) for l in open(steps)]
    if LIMIT:
        J = J[:LIMIT]
    G, EC, FO, rows, arms_rows = [], [], [], [], []
    t0 = time.time()
    skipped = 0
    worst_sum = 0.0
    for c, j in enumerate(J):
        P = {a: prepare(j, a) for a in ARMS}
        if any(v is None for v in P.values()):
            skipped += 1
            continue
        tails = [P[a]["ids"][P[a]["after"]:] for a in ARMS]
        if not (tails[0] == tails[1] == tails[2]):
            skipped += 1
            continue
        heads = [P[a]["ids"][:P[a]["first"]] for a in ARMS]
        if not (heads[0] == heads[1] == heads[2]):
            skipped += 1
            continue
        tail_len = len(tails[0])
        for a in ARMS:
            p = P[a]
            ids = p["ids"]
            idx, ks = [], []
            for k in KS:
                if k == -1:
                    i = p["first"] - 1
                elif k == 0:
                    i = p["after"] - 1
                elif k <= tail_len:
                    i = p["after"] + k - 1
                else:
                    continue
                idx.append(i)
                ks.append(k)
            IDX["i"] = torch.tensor(idx, device=DEV)
            with torch.inference_mode():
                ti = torch.tensor([ids], device=DEV)
                hs = m.model(input_ids=ti).last_hidden_state[0]
                lpn = m.lm_head(hs[p["first"] - 1:p["after"] - 1]).float().log_softmax(-1) / np.log(2)
                ss = -lpn.gather(1, ti[0, p["first"]:p["after"]][:, None])[:, 0].sum()
                lo = m.lm_head(hs[IDX["i"]]).float()
                sel = lo.log_softmax(-1) / np.log(2)
                ent = -(sel.exp() * sel).sum(-1)
                top = lo.topk(2, dim=-1).values
                nextid = torch.tensor([ids[i + 1] if i + 1 < len(ids) else 0 for i in idx], device=DEV)
                nextsurp = -sel.gather(1, nextid[:, None])[:, 0]
                Wr = torch.stack([got[n] for n in order], 1)
                gram = Wr @ Wr.transpose(1, 2)
                summed = got["x0"] + sum(got[f"op{L}"] for L in range(N)) + sum(got[f"ffn{L}"] for L in range(N - 1))
                worst_sum = max(worst_sum, float(((summed - got["xlast"]).norm(dim=-1) / got["xlast"].norm(dim=-1)).max()))
                ec = []
                for L in range(N):
                    g = got[f"xt{L}"] @ W13[L][0].T
                    e = got[f"xt{L}"] @ W13[L][1].T
                    mm = F.silu(g)
                    ec.append(0.5 * ((g <= 0).float() * mm.abs() * e ** 2).sum(-1))
                ec = torch.stack(ec, 1)
            G.append(gram.cpu().numpy()[:, iu[0], iu[1]].astype(np.float32))
            EC.append(ec.cpu().numpy().astype(np.float32))
            FO.append(torch.stack([ent, top[:, 0] - top[:, 1], sel.max(-1).values.exp(), nextsurp], 1).cpu().numpy().astype(np.float32))
            for q, k in enumerate(ks):
                rows.append(dict(step=j["step"], id=j["id"], split=j["split"], arm=a, k=k, seen_before=j["seen_before"], num_surprise=float(ss), tok=int(ids[idx[q]])))
        if c % 50 == 0:
            print(c, len(J), "skipped", skipped, "sum error", round(worst_sum, 5), round(time.time() - t0), flush=True)
    np.savez(prefix + "-x.npz", G=np.concatenate(G), EC=np.concatenate(EC), FO=np.concatenate(FO), order=np.array(order))
    with open(prefix + "-rows.jsonl", "w") as w:
        for r in rows:
            w.write(json.dumps(r) + "\n")
    print("BROKEN_DONE steps", len(J) - skipped, "skipped", skipped, "sum error", round(worst_sum, 5), round(time.time() - t0), flush=True)


main()
