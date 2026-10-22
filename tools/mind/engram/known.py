"""Engram, the "do I know this?" signal. Is there something in his own brain, on the word where he is about to answer,
that says whether a memory for the question is there (feeling of knowing / familiarity)? Others found one in Gemma 2
(Ferrando et al. 2025: known vs unknown entity pieces, middle layers) and in Claude (Anthropic 2025: a default "can't
answer" switched off by "known" features; misfires = confabulation).

His thoughts come from engine/kv-check/build/bricks --watch ... --prompt-only (his live brain, adapter on): the last
prompt word's stream going into layers 8..42, for cases where the right memory is there or not, with the SAME
questions on both sides (pile of 5 with the right one vs 4 that don't fit; one memory vs nothing), so the question's
wording can't give it away.

  python known.py --dir DIR [--sae12 sae_L12.pt]
  python known.py --cases2 DIR2 --dump DUMP --sae8 sae_L8.pt         # test 2 cases (+ bricks-faint.f32)
  python known.py --test2 DIR2 --dir DIR                              # score them with DIR's direction

Test 2 pulls "a memory is there" apart from "I'm about to say I don't know":
  faint     bricks with 16 or 8 pieces a word before the question: he often says he doesn't know while the right
            answer still ranks first (tip of the tongue). Does the signal follow what he half-remembers?
  telling   his turn opens with You told me: " and the memory (words, bricks, another moment's bricks, or nothing)
            was before the question: he has to say something either way, so "I don't know" isn't on the table.
            With nothing there he has to make it up. Does the signal still split them?

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


def cases2(a):
    import torch
    from livebricks import END
    from livefit import load
    from moments import moments, prefix
    from sae import load_sae
    out = Path(a.cases2)
    out.mkdir(parents=True, exist_ok=True)
    ms = moments()
    _, _, spans = load(a.dump)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sae = load_sae(a.sae8, dev)
    rows, at = [], {}
    for k in (sae.k, 16, 8):
        for i, h in enumerate(spans):
            at[k, i] = sum(len(r) for r in rows)
            with torch.no_grad():
                rows.append(sae.roundtrip(torch.from_numpy(h).to(dev), k).cpu().numpy())
    np.concatenate(rows).astype(np.float32).tofile(out / "bricks-faint.f32")

    def bricks(j, k):
        return dict(slot=ms[j]["text"], n=len(spans[j]), brick=at[k, j])

    with open(out / "cases.jsonl", "w") as f:
        for i, m in enumerate(ms):
            q = "\n\n" + m["question"] + END
            for way, k in (("saeple8k16", 16), ("saeple8k8", 8)):
                f.write(json.dumps(dict(parts=[dict(text=prefix(m["ctx_b"])), bricks(i, k), dict(text=q)],
                                        cands=m["cands"], meta=dict(set="faint", way=way, i=i, kind=m["kind"],
                                                                    question=m["question"], want=m["answer"]))) + "\n")
            for way, mem in (("text", dict(text=m["text"])), ("saeple8", bricks(i, sae.k)),
                             ("saepleswap8", bricks(m["swap"], sae.k)), ("none", None)):
                parts = [dict(text=prefix(m["ctx_b"]))] + ([mem] if mem else []) + [dict(text=q + 'You told me: "')]
                want = ms[m["swap"]]["answer"] if way == "saepleswap8" else m["answer"]
                f.write(json.dumps(dict(parts=parts, cands=m["cands"], meta=dict(
                    set="telling", way=way, i=i, kind=m["kind"], question=m["question"], want=want))) + "\n")
    print(f"wrote {out / 'cases.jsonl'} and {out / 'bricks-faint.f32'}")


def test2(a):
    from livebricks import has
    from moments import REFUSED
    d1, d2 = Path(a.dir), Path(a.test2)
    layers = [8, 12, 16, 20, 24, 30, 36, 42]
    r1 = [json.loads(l)["meta"] for l in open(d1 / "watched.jsonl")]
    X1 = np.fromfile(d1 / "thoughts.f32", dtype=np.float32).reshape(len(r1), len(layers), -1)
    s1 = np.array([m["set"] for m in r1])
    y1 = np.array([1 if (s in ("pile5", "same2") or (s == "single" and m["way"] in THERE)) else 0
                   for s, m in zip(s1, r1)])
    train = np.isin(s1, ["pile5", "gate4", "single"])
    r2 = [json.loads(l) for l in open(d2 / "answers.jsonl")]
    X2 = np.fromfile(d2 / "thoughts.f32", dtype=np.float32).reshape(len(r2), len(layers), -1)
    m2 = [r["meta"] for r in r2]
    said = [r["said"].strip() for r in r2]
    ok = np.array([has(t, m["want"]) for t, m in zip(said, m2)])
    refused = np.array([(not o) and REFUSED.search(t) is not None for o, t in zip(ok, said)])
    top = np.array([max(r["scores"], key=r["scores"].get) == m["want"] for r, m in zip(r2, m2)])
    st = np.array([m["set"] for m in m2])
    way = np.array([m["way"] for m in m2])
    res = {"layers": {}}
    print("telling (his turn opens with You told me: \", he must say something): memory there vs nothing")
    print(f"{'layer':>6} {'words/none':>11} {'bricks/none':>12} {'other bricks/none':>18}   faint refusals: ranked "
          f"right vs not (mean score)")
    for li, L in enumerate(layers):
        x1, x2 = unit(X1[:, li]), unit(X2[:, li])
        w = direction(x1[train], y1[train])
        mid = (x1[train & (y1 == 1)] @ w).mean() / 2 + (x1[train & (y1 == 0)] @ w).mean() / 2
        sc = x2 @ w - mid                       # > 0 reads as "there" (midway between the first test's two sides)
        none = sc[(st == "telling") & (way == "none")]
        row = {f"telling_{v}": auroc(sc[(st == "telling") & (way == v)], none) for v in ("text", "saeple8", "saepleswap8")}
        fr = (st == "faint") & refused
        row["faint_refused_ranked_right"] = float(sc[fr & top].mean()) if (fr & top).any() else None
        row["faint_refused_ranked_wrong"] = float(sc[fr & ~top].mean()) if (fr & ~top).any() else None
        row["faint_refused_auroc_ranked"] = auroc(sc[fr & top], sc[fr & ~top]) if (fr & top).any() and (fr & ~top).any() else None
        row["faint_said_right"] = float(sc[(st == "faint") & ok].mean()) if ((st == "faint") & ok).any() else None
        row["telling_none_above_mid"] = int((none > 0).sum())
        res["layers"][L] = row
        fa = row["faint_refused_auroc_ranked"]
        print(f"{L:>6} {row['telling_text']:11.3f} {row['telling_saeple8']:12.3f} {row['telling_saepleswap8']:18.3f}"
              f"   {row['faint_refused_ranked_right'] if row['faint_refused_ranked_right'] is not None else float('nan'):+.3f}"
              f" vs {row['faint_refused_ranked_wrong'] if row['faint_refused_ranked_wrong'] is not None else float('nan'):+.3f}"
              f" (AUROC {fa if fa is not None else float('nan'):.3f})")
    counts = {}
    for v in ("saeple8k16", "saeple8k8"):
        sel = (st == "faint") & (way == v)
        counts[v] = dict(n=int(sel.sum()), said_right=int((sel & ok).sum()), refused=int((sel & refused).sum()),
                         refused_but_ranked_right=int((sel & refused & top).sum()))
    for v in ("text", "saeple8", "saepleswap8", "none"):
        sel = (st == "telling") & (way == v)
        counts["telling_" + v] = dict(n=int(sel.sum()), said_right=int((sel & ok).sum()))
    res["counts"] = counts
    print(json.dumps(counts))
    with open(d2 / "graded.jsonl", "w") as f:
        for j, m in enumerate(m2):
            f.write(json.dumps(dict(**m, said=said[j], said_right=bool(ok[j]), refused=bool(refused[j]),
                                    ranked_right=bool(top[j]),
                                    score={L: float(unit(X2[j, li]) @ direction(unit(X1[:, li])[train], y1[train]))
                                           for li, L in enumerate(layers)})) + "\n")
    json.dump(res, open(d2 / "known2.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--sae12", default="")
    ap.add_argument("--cases2", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--sae8", default="")
    ap.add_argument("--test2", default="")
    a = ap.parse_args()
    if a.cases2:
        return cases2(a)
    if a.test2:
        return test2(a)
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
