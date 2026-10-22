"""Engram, live brain step 2: bricks in his real engine. The 64 moments of rung0.py, now on his live brain (grid4 +
his talk adapter) in llama.cpp (engine/kv-check/build/bricks), in the same three orders as rung0.py: before the
question, after it, and at the start of his own turn as You told me: "...".

Each memory was read once in its own chat (context A) by his live brain, and its stream going into layer 8 was kept
(livefit.py's layer dump). Ways, as in rung0.py:
  text          the moment's words, read fresh                      none   nothing (guessing)
  ple8          the kept layer-8 stream as is + name tags (the ceiling for bricks)
  saeple8       bricks: the stream after a trip through the 64 fragment pieces a word + name tags
  saeple8k32    bricks with 32 pieces a word
  saepleswap8   another moment's bricks (cheat check: the answer should follow the memory he was handed)

  python livebricks.py write --dump DIR/grid4-adapter --sae sae_L8.pt --out DIR2   # cases.jsonl + bricks.f32
  engine/kv-check/build/bricks BRAIN DIR2/cases.jsonl DIR2/bricks.f32 DIR2/answers.jsonl --layer 8 --lora ADAPTER
  python livebricks.py grade --out DIR2
  python livebricks.py piles --dump DIR/grid4-adapter --out DIR2               # piles.jsonl (same bricks.f32)
  engine/kv-check/build/bricks BRAIN DIR2/piles.jsonl DIR2/bricks.f32 DIR2/answers-piles.jsonl --layer 8 --lora ADAPTER
  python livebricks.py grade-piles --out DIR2

Piles (reader.py's, now with bricks): each memory rebuilt in the new chat one after another right before the
question, so a later one is rebuilt after an earlier one and (from layer 8 up) can see it, unlike saved photos.
  same2   an older memory of the same kind, then the right (newer) one: red table, then purple table
  same2_recall   the same two, each at the start of his own turn as You told me: "..."
  pile5   the right memory among 4 of other kinds         gate4   4 memories of other kinds, not the right one
"""
import argparse, collections, json, re
from pathlib import Path

import numpy as np

from moments import KINDS, PER_KIND, REFUSED, moments, prefix

ORDERS = ("before", "after", "recall")
END = "<turn|>\n<|turn>model\n"


def has(text, value):   # rung0.py's
    return re.search(r"\b" + re.escape(value.lower()) + r"\b", text.lower()) is not None


def around(m, order):
    """(chat B up to the memory, chat B after it), as rung0.py"""
    return {"before": ("", "\n\n" + m["question"] + END),
            "after": (m["question"] + "\n\n", END),
            "recall": (m["question"] + END + 'You told me: "', '"\n\n')}[order]


def offsets(spans):
    """each way's first brick row per moment in bricks.f32: all ple8 rows, then saeple8, then saeple8k32"""
    at, r = {}, 0
    for way in ("ple8", "saeple8", "saeple8k32"):
        for i, h in enumerate(spans):
            at[way, i] = r
            r += len(h)
    return at


def write(a):
    import torch
    from livefit import load
    from sae import load_sae
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ms = moments()
    meta, _, spans = load(a.dump)
    assert len(spans) == len(ms) and meta["layer"] == 8 and meta["adapter"], "want his live brain's layer-8 dump"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sae = load_sae(a.sae, dev)
    kinds = {"ple8": None, "saeple8": sae.k, "saeple8k32": 32}
    rows, at = [], offsets(spans)
    for way, k in kinds.items():
        for i, h in enumerate(spans):
            assert at[way, i] == sum(len(r) for r in rows)
            if k is None:
                rows.append(h)
            else:
                with torch.no_grad():
                    rows.append(sae.roundtrip(torch.from_numpy(h).to(dev), k).cpu().numpy())
    np.concatenate(rows).astype(np.float32).tofile(out / "bricks.f32")

    def slot(j, way):
        return dict(slot=ms[j]["text"], n=len(spans[j]), brick=at[way, j])

    with open(out / "cases.jsonl", "w") as f:
        m = ms[0]   # the splice check: moment 0 in its own chat, the question right after it
        f.write(json.dumps(dict(check=True, parts=[dict(text=prefix(m["ctx_a"])), dict(slot=m["text"], n=len(spans[0])),
                                                   dict(text="\n\n" + m["question"] + END)])) + "\n")
        for i, m in enumerate(ms):
            ways = {"text": dict(text=m["text"]), "none": None, "ple8": slot(i, "ple8"), "saeple8": slot(i, "saeple8"),
                    "saeple8k32": slot(i, "saeple8k32"), "saepleswap8": slot(m["swap"], "saeple8")}
            for order in ORDERS:
                lead, tail = around(m, order)
                for way, mem in ways.items():
                    parts = [dict(text=prefix(m["ctx_b"]))] + ([dict(text=lead)] if lead else []) + \
                            ([mem] if mem else []) + [dict(text=tail)]
                    want = ms[m["swap"]]["answer"] if way == "saepleswap8" else m["answer"]
                    f.write(json.dumps(dict(parts=parts, cands=m["cands"], meta=dict(
                        i=i, kind=m["kind"], way=way if order == "before" else f"{way}_{order}", order=order,
                        moment=m["text"], question=m["question"], want=want, own=m["answer"]))) + "\n")
    print(f"wrote {out / 'cases.jsonl'} and {out / 'bricks.f32'}")


PILES = ("same2", "same2_recall", "pile5", "gate4")


def piles(a):
    from livefit import load
    from reader import pile_for
    out = Path(a.out)
    ms = moments()
    _, _, spans = load(a.dump)
    at = offsets(spans)
    with open(out / "piles.jsonl", "w") as f:
        for way in PILES:
            for i, m in enumerate(ms):
                pile = pile_for(way.split("_")[0], i, ms)
                for mem in ("text", "saeple8"):
                    def one(j):
                        return dict(text=ms[j]["text"]) if mem == "text" else \
                            dict(slot=ms[j]["text"], n=len(spans[j]), brick=at[mem, j])
                    parts = [dict(text=prefix(m["ctx_b"]))]
                    if way.endswith("recall"):
                        parts.append(dict(text=m["question"] + END))
                        for j in pile:
                            parts += [dict(text='You told me: "'), one(j), dict(text='"\n\n')]
                    else:
                        parts += [one(j) for j in pile] + [dict(text="\n\n" + m["question"] + END)]
                    f.write(json.dumps(dict(parts=parts, cands=m["cands"], meta=dict(
                        i=i, kind=m["kind"], pile_way=way, way=mem, pile=pile, question=m["question"],
                        want=m["answer"]))) + "\n")
    print(f"wrote {out / 'piles.jsonl'}")


def grade_piles(a):
    from reader import grade as verdict
    out = Path(a.out)
    ms = moments()
    rows = [json.loads(l) for l in open(out / "answers-piles.jsonl") if "meta" in l]
    for r in rows:
        m = r["meta"]
        r.update(m, said=r["said"].strip(), pile_text=[ms[j]["text"] for j in m["pile"]],
                 verdict=verdict(m["pile_way"].split("_")[0], r["said"].strip(), m["i"], m["pile"], ms))
        del r["meta"]
    with open(out / "graded-piles.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {}
    for pw in dict.fromkeys(r["pile_way"] for r in rows):
        summary[pw] = {w: dict(sorted(collections.Counter(r["verdict"] for r in rows
                                                          if r["pile_way"] == pw and r["way"] == w).items()))
                       for w in ("text", "saeple8")}
        print(pw, json.dumps(summary[pw]))
    json.dump(summary, open(out / "summary-piles.json", "w"), indent=1)


def grade(a):
    out = Path(a.out)
    lines = [json.loads(l) for l in open(out / "answers.jsonl")]
    for l in lines:
        if "check" in l:
            print("splice check:", json.dumps(l["check"]))
    recs = []
    for l in lines:
        if "meta" not in l:
            continue
        m, said, sc = l["meta"], l["said"].strip(), l["scores"]
        top = max(sc, key=sc.get)
        p = np.exp(np.array(list(sc.values())) - max(sc.values()))
        recs.append(dict(**m, said=said, said_right=has(said, m["want"]), said_own=has(said, m["own"]),
                         refused=not has(said, m["want"]) and REFUSED.search(said) is not None,
                         top=top, top_right=top == m["want"], p_right=float(p[list(sc).index(m["want"])] / p.sum())))
    with open(out / "graded.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summary = {}
    print(f"\n{'way':22} {'said right':>10} {'refused':>8} {'ranked 1st':>10} {'p(right)':>9}   (of 64)")
    for w in dict.fromkeys(r["way"] for r in recs):
        rs = [r for r in recs if r["way"] == w]
        s = {k: sum(r[k] for r in rs) for k in ("said_right", "refused", "top_right")}
        s["p_right"] = sum(r["p_right"] for r in rs) / len(rs)
        s["by_kind"] = {k: sum(r["top_right"] for r in rs if r["kind"] == k) for k in KINDS}
        summary[w] = dict(n=len(rs), **s)
        print(f"{w:22} {s['said_right']:10d} {s['refused']:8d} {s['top_right']:10d} {s['p_right']:9.3f}")
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--dump", required=True)
    w.add_argument("--sae", required=True)
    w.add_argument("--out", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--out", required=True)
    p = sub.add_parser("piles")
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    gp = sub.add_parser("grade-piles")
    gp.add_argument("--out", required=True)
    a = ap.parse_args()
    {"write": write, "grade": grade, "piles": piles, "grade-piles": grade_piles}[a.cmd](a)
