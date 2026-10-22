"""Which details of a rich memory fade first when his bricks get fewer? (Immy: closer to a human mind than a number.)

Each made-up memory from richbank.py (who, what happened, where, when, a number, a quote, a side detail, a feeling, a
second person) is read once in its own chat A and saved; in chat B he is asked nine questions about it, one per
detail, with the memory put back in one of these ways:

  text         the words read fresh (ceiling)        none        nothing (floor)
  kv           the saved KV cache (the photo)
  saeple8kN    his layer-8 pieces + name tags, rebuilt from layer 8 up, N pieces a word (64 = all, 32, 16, 8)
  sae8         all 64 pieces, no name tags
  saepleswap8  another memory's pieces + tags (cheat check: the answers must follow that memory)

Orders: before (memory, then question) for every way; recall (question, then his turn opens `You told me: "..."`)
for text, kv, saeple8, saeple8k32. Greedy answers, graded by key words (a pointer; every answer is read by hand).

  python richtest.py --model DIR --memories memories.json --sae SAE_L8.pt --out DIR
  python richtest.py --memories memories.json --out DIR --regrade     # grade saved answers again (no GPU)
"""
import argparse, json, re, time
from pathlib import Path

from moments import REFUSED, prefix

L = 8
KS = [None, 32, 16, 8]
TYPES = ["who", "gist", "where", "when", "number", "quote", "side", "feeling", "bind"]
# questions of the first set that asked something else than meant (answers to them are right readings, not misses):
# "Whose workplace had the burst pipe?" is answered with the workplace; "What did my coworker win?" with the prize
BAD_QUESTION = {("flood", "who")}
UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
ORDS = "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split()
VALUE = {**{w: i for i, w in enumerate(UNITS)}, **{w: 20 + 10 * i for i, w in enumerate(TENS)},
         **{w: i + 1 for i, w in enumerate(ORDS)}, **{w + "th": i + 13 for i, w in enumerate(UNITS[13:])},
         **{w[:-1] + "ieth": 20 + 10 * i for i, w in enumerate(TENS)}}
NUMWORD = "|".join(sorted(VALUE, key=len, reverse=True) + ["hundred"])
NUMRUN = re.compile(rf"\b(?:{NUMWORD})(?:[\s-]+(?:and[\s-]+)?(?:{NUMWORD}))*\b", re.I)


def number_words(s):
    """spelled-out numbers -> digits ("One hundred sixteen" -> 116, "twenty-fifth" -> 25)"""
    def value(run):
        total = 0
        for w in re.findall(r"[a-z]+", run.group(0).lower()):
            if w == "hundred":
                total = max(total, 1) * 100
            elif w != "and":
                total += VALUE[w]
        return str(total)
    return NUMRUN.sub(value, s)


def hit(said, key, typ):
    if typ == "number":
        return re.search(rf"(?<!\d){re.escape(key)}(?!\d)", number_words(said)) is not None
    return re.search(r"\b" + re.escape(key.lower()), said.lower()) is not None


def verdict(said, q, m, other=None):
    """right / other person / refused / wrong / bad question, for question q about memory m"""
    if (m["kind"], q["type"]) in BAD_QUESTION:
        return "bad question"
    need = all if q["type"] == "quote" else any
    if need(hit(said, k, q["type"]) for k in q["keys"]):
        return "right"
    if (m["kind"], q["type"]) == ("trivia", "gist") and hit(said, m["slots"]["num"], "number"):
        return "right"                   # "What did my coworker win?" -> the prize
    if any(hit(said, k, q["type"]) for k in (q["other"] if other is None else other)):
        return "other person"
    if REFUSED.search(said):
        return "refused"
    return "wrong"


def grade(rows, ms):
    """verdicts for saved answers (the cheat-check rows against the other memory's details)"""
    byi = {m["i"]: m for m in ms}
    for r in rows:
        m = byi[r["i"]]
        if r["way"] == "saepleswap8":
            m = byi[m["swap"]]
        q = next(x for x in m["questions"] if x["type"] == r["type"])
        r["verdict"] = verdict(r["said"], q, m)
    return rows


def summarize(rows, out):
    summary = {}
    for key in dict.fromkeys((r["way"], r["order"]) for r in rows):
        rs = [r for r in rows if (r["way"], r["order"]) == key and r["verdict"] != "bad question"]
        summary[f"{key[0]}_{key[1]}"] = {t: {v: sum(r["verdict"] == v for r in rs if r["type"] == t)
                                             for v in ("right", "refused", "wrong", "other person")}
                                         | {"n": sum(r["type"] == t for r in rs)} for t in TYPES}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(f"\n{'way':24}" + "".join(f"{t:>8}" for t in TYPES) + "   (right; who is of 56: one question was bad)")
    for name, by in summary.items():
        print(f"{name:24}" + "".join(f"{by[t]['right']:>8}" for t in TYPES))


def pair_swaps(ms):
    for m in ms:                         # swap partner: the next memory of the same kind
        same = [x["i"] for x in ms if x["kind"] == m["kind"]]
        m["swap"] = same[(same.index(m["i"]) + 1) % len(same)]


def run(a, ms, out):
    import torch
    import rung0
    from rung0 import N_KV, Rig
    from sae import load_sae
    rung0.MAX_NEW = 24                   # quotes in his own turn start with "So, she was like, ..." (12 cut them)
    rig, sae = Rig(a.model), load_sae(a.sae)
    saved = {}
    for m in ms:                         # read each memory once in its own chat A
        span = rig.ids(m["text"])
        mem, _ = rig.capture(rig.ids(prefix(m["ctx_a"])), span)
        with torch.no_grad():
            bricks = {k: {**mem, "h": {**mem["h"], L: sae.roundtrip(mem["h"][L], k)}} for k in KS}
        saved[m["i"]] = dict(span=span, mem=mem, bricks=bricks)

    def ways(m):
        sv, sw = saved[m["i"]], saved[m["swap"]]
        n, n_sw = len(sv["span"]), len(sw["span"])
        out = [("text", None, None), ("none", 0, None),
               ("kv", n, lambda s: dict(s=s, n=n, mem=sv["mem"], kv_layers=list(range(N_KV))))]
        for k in KS:
            out.append((f"saeple8k{k or 64}", n,
                        lambda s, k=k: dict(s=s, n=n, mem=sv["bricks"][k], L=L, block=True, ple=True)))
        out.append(("sae8", n, lambda s: dict(s=s, n=n, mem=sv["bricks"][None], L=L, block=True)))
        out.append(("saepleswap8", n_sw, lambda s: dict(s=s, n=n_sw, mem=sw["bricks"][None], L=L, block=True,
                                                         ple=True)))
        return out

    RECALL = {"text", "kv", "saeple8k64", "saeple8k32"}
    rows_out = open(out / "rich.jsonl", "w")
    t0 = time.time()
    for m in ms:
        sv = saved[m["i"]]
        pre = rig.ids(prefix(m["ctx_b"]))
        for q in m["questions"]:
            around = {"before": ([], rig.ids("\n\n" + q["q"] + "<turn|>\n<|turn>model\n")),
                      "recall": (rig.ids(q["q"] + "<turn|>\n<|turn>model\n" + 'You told me: "'), rig.ids('"\n\n'))}
            for name, slot_n, build in ways(m):
                for order in ("before", "recall"):
                    if order == "recall" and name not in RECALL:
                        continue
                    lead, tail = around[order]
                    slot = sv["span"] if name == "text" else [rig.pad] * slot_n
                    said = rig.answer(pre + lead + slot + tail, build(len(pre) + len(lead)) if build else None)
                    rows_out.write(json.dumps(dict(i=m["i"], kind=m["kind"], type=q["type"], way=name, order=order,
                                                   q=q["q"], keys=q["keys"], said=said)) + "\n")
        rows_out.flush()
        print(f"{m['i'] + 1}/{len(ms)} memories done, {time.time() - t0:.0f} s", flush=True)
    rows_out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--memories", required=True)
    ap.add_argument("--sae", help="the layer-8 fragment finder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regrade", action="store_true", help="grade the answers already in OUT (no GPU)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ms = json.load(open(a.memories))
    pair_swaps(ms)
    if not a.regrade:
        run(a, ms, out)
    rows = grade([json.loads(l) for l in open(out / "rich.jsonl")], ms)
    with open(out / "graded.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summarize(rows, out)


if __name__ == "__main__":
    main()
