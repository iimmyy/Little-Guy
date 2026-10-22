"""Engram, the "do I know this?" signal. Is there something in his own brain, on the word where he is about to answer,
that says whether a memory for the question is there (feeling of knowing / familiarity)? Others found one in Gemma 2
(Ferrando et al. 2025: known vs unknown entity pieces, middle layers) and in Claude (Anthropic 2025: a default "can't
answer" switched off by "known" features; misfires = confabulation).

His thoughts come from engine/kv-check/build/bricks --watch ... --prompt-only (his live brain, adapter on): the last
prompt word's stream going into layers 8..42, for cases where the right memory is there or not, with the SAME
questions on both sides (pile of 5 with the right one vs 4 that don't fit; one memory vs nothing), so the question's
wording can't give it away.

  python known.py --dir DIR [--sae12 sae_L12.pt]

Per layer: the direction from "not there" to "there" (difference of means), learned on the piles and tested on the
single-memory cases it never saw (and the other way round), and learned on 7 of the 8 kinds and tested on the 8th;
AUROC (0.5 = coin flip, 1 = perfect). With --sae12, the layer-12 finder's pieces that best tell the two apart.
"""
import argparse, json
from pathlib import Path

import numpy as np

from moments import KINDS

THERE = {"text", "saeple8", "saepleswap8"}


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1.0
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def direction(x, y):
    d = x[y == 1].mean(0) - x[y == 0].mean(0)
    return d / np.linalg.norm(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sae12", default="")
    a = ap.parse_args()
    d = Path(a.dir)
    rows = [json.loads(l) for l in open(d / "watched.jsonl")]
    layers = [8, 12, 16, 20, 24, 30, 36, 42]
    X = np.fromfile(d / "thoughts.f32", dtype=np.float32).reshape(len(rows), len(layers), -1)
    meta = [r["meta"] for r in rows]
    setname = np.array([m["set"] for m in meta])
    mem = np.array([m["way"] for m in meta])
    kind = np.array([m["kind"] for m in meta])
    y = np.array([1 if (s in ("pile5", "same2") or (s == "single" and w in THERE)) else 0
                  for s, w in zip(setname, mem)])
    piles = np.isin(setname, ["pile5", "gate4"])
    single = setname == "single"
    res = {"n": len(rows), "layers": {}}
    print(f"{'into layer':>10} {'piles->single':>14} {'single->piles':>14} {'leave a kind out':>17}")
    for li, L in enumerate(layers):
        x = unit(X[:, li])
        out = {}
        for name, tr, te in (("piles->single", piles, single), ("single->piles", single, piles)):
            w = direction(x[tr], y[tr])
            s = x[te] @ w
            out[name] = auroc(s[y[te] == 1], s[y[te] == 0])
        held = []
        for k in KINDS:
            tr, te = (piles | single) & (kind != k), (piles | single) & (kind == k)
            w = direction(x[tr], y[tr])
            s = x[te] @ w
            held.append(auroc(s[y[te] == 1], s[y[te] == 0]))
        out["leave_kind_out"] = float(np.mean(held))
        out["leave_kind_out_worst"] = float(np.min(held))
        # the bricks alone (training on everything else that isn't the case's kind is above; here: bricks vs none)
        w = direction(x[piles], y[piles])
        s = x @ w
        b = single & np.isin(mem, ["saeple8", "none"])
        out["bricks_vs_nothing"] = auroc(s[b & (y == 1)], s[b & (y == 0)])
        res["layers"][L] = out
        print(f"{L:>10} {out['piles->single']:14.3f} {out['single->piles']:14.3f} {out['leave_kind_out']:11.3f}"
              f" (worst {out['leave_kind_out_worst']:.3f})")
    # where the made-up answers and the same-kind pairs fall, on the best layer's piles direction
    best = max(layers, key=lambda L: res["layers"][L]["piles->single"])
    li = layers.index(best)
    x = unit(X[:, li])
    w = direction(x[piles], y[piles])
    s = x @ w
    thr = (s[piles & (y == 1)].mean() + s[piles & (y == 0)].mean()) / 2
    res["best_layer"] = best
    res["same2_above_middle"] = int((s[setname == "same2"] > thr).sum())
    res["same2_n"] = int((setname == "same2").sum())
    print(f"best layer {best}: same-kind pairs read as 'there' {res['same2_above_middle']}/{res['same2_n']}")
    ans = {}
    for f in ("answers-piles.jsonl",):
        p = Path(a.dir).parent.parent / "livebricks" / "20261022-0320" / f
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                m = r["meta"]
                ans[(m["pile_way"], m["way"], m["i"])] = r["said"].strip()
    flagged = []
    for j, m in enumerate(meta):
        if m["set"] == "gate4":
            said = ans.get(("gate4", m["way"], m["i"]), "")
            flagged.append(dict(way=m["way"], i=m["i"], question=m["question"], said=said, score=float(s[j]),
                                above_middle=bool(s[j] > thr)))
    flagged.sort(key=lambda r: -r["score"])
    print("none-fits cases that look most like 'there' (top 6):")
    for r in flagged[:6]:
        print(f"  {r['score']:+.3f} {'ABOVE' if r['above_middle'] else 'below'} {r['way']:8} {r['said'][:60]!r}")
    res["gate4_most_there"] = flagged[:10]
    if a.sae12:
        import torch
        from sae import load_sae
        sae = load_sae(a.sae12, "cpu")
        li = layers.index(12)
        with torch.no_grad():
            v, i = sae.encode(torch.from_numpy(X[:, li]).float() / sae.scale)
        z = torch.zeros(len(rows), sae.W_dec.shape[0]).scatter(-1, i, v).numpy()
        on = z > 0
        sep = on[piles & (y == 1)].mean(0) - on[piles & (y == 0)].mean(0)
        top = np.argsort(-np.abs(sep))[:8]
        res["sae12_pieces"] = []
        print("layer-12 pieces that tell them apart (learned on piles; firing rate there / not there, on single):")
        for t in top:
            fs = [on[single & (y == 1), t].mean(), on[single & (y == 0), t].mean()]
            res["sae12_pieces"].append(dict(piece=int(t), piles_sep=float(sep[t]), single_there=float(fs[0]),
                                            single_not=float(fs[1])))
            print(f"  piece {t:5d}  piles {sep[t]:+.2f}   single: there {fs[0]:.2f}  not there {fs[1]:.2f}")
    json.dump(res, open(d / "known.json", "w"), indent=1)


if __name__ == "__main__":
    main()
