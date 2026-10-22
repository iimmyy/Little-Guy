"""Engram: bent memories (Loftus-style misinformation). Can anything tell a true memory from a false one?

His strength signal (known.py, layer 24) says how strongly a memory clicks, not whether it's true. Here one detail of
each of the 64 rung-0 moments is changed (the answer swapped for another candidate that takes the same number of
word pieces), and the memory comes back four ways, before the question, on his live brain + adapter:
  genuine      bricks and name tags of the real moment
  tags_bent    the real moment's bricks, the bent moment's name tags (the exact word swapped, the gist not: what a
               bad merge in sleep could do)
  bricks_bent  the bent moment's bricks (read in its own chat A), the real name tags
  both_bent    bricks and tags of the bent moment: a false memory encoded as if it were real
  words_bent   the bent moment's words (reference)
Measured: what he says (the real detail, the bent one, both), the strength signal, and whether bricks and tags
disagree: each brick row against a fresh reading of the tag words alone at layer 8 (both through the finder). The
blind check (it doesn't know which word changed): the biggest dip below the memory's own typical match, skipping its
first 3 words (they always differ a little: the memory was read after a chat, the fresh reading after nothing).

  python bent.py write --tokenizer tokenizer.json --out DIR          # bent.json, chat-a.jsonl, fresh.jsonl
  layer-dump BRAIN DIR/chat-a.jsonl DIR/bent-a 8 --lora ADAPTER      # bent moments read in their own chat A
  layer-dump BRAIN DIR/fresh.jsonl DIR/fresh 8 --lora ADAPTER        # real and bent words read alone
  python bent.py cases --dump LIVEFIT/grid4-adapter --sae sae_L8.pt --out DIR      # bricks.f32, cases.jsonl
  bricks BRAIN DIR/cases.jsonl DIR/bricks.f32 DIR/answers.jsonl --layer 8 --lora ADAPTER \\
         --watch 24 --watch-out DIR/thoughts.f32
  python bent.py grade --out DIR --known KNOWN_DIR                   # KNOWN_DIR: known.py's first test
"""
import argparse, json, random
from pathlib import Path

import numpy as np

from moments import moments, prefix

WAYS = ("genuine", "tags_bent", "bricks_bent", "both_bent")
SPARE = ["Amira", "Anouk", "Zainab", "Leilani", "Saoirse", "Thandiwe", "Ottilie", "Kofi"]   # when no candidate fits
ALONE = "<bos><|turn>user\n"


def write(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    n = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bank = []
    for i, m in enumerate(moments()):
        t, v = m["text"], m["answer"]
        assert t.count(v) == 1, (i, v, t)
        rng = random.Random(f"bent-{i}")
        cands = [c for c in m["cands"] if c not in (v, m["distractor"]) and c not in t]
        rng.shuffle(cands)
        cands += [c for c in SPARE if c not in t] if m["kind"] in ("puppy", "cousin") else []
        b = next((c for c in cands if n(t.replace(v, c)) == n(t)), None)
        assert b, f"moment {i}: no same-length swap"
        bank.append(dict(i=i, kind=m["kind"], text=t, bent_text=t.replace(v, b), answer=v, bent=b,
                         question=m["question"]))
    json.dump(bank, open(out / "bent.json", "w"), indent=1)
    ms = moments()
    with open(out / "chat-a.jsonl", "w") as f:
        for r in bank:
            f.write(json.dumps(dict(a=prefix(ms[r["i"]]["ctx_a"]), moment=r["bent_text"])) + "\n")
    with open(out / "fresh.jsonl", "w") as f:
        for r in bank:
            for t in (r["text"], r["bent_text"]):
                f.write(json.dumps(dict(a=ALONE, moment=t)) + "\n")
    print(f"wrote {len(bank)} bent moments to {out}")


def cases(a):
    import torch
    from livebricks import END
    from livefit import load
    from sae import load_sae
    out = Path(a.out)
    bank = json.load(open(out / "bent.json"))
    ms = moments()
    _, _, real = load(a.dump)
    _, _, bent = load(str(out / "bent-a"))
    _, _, fresh = load(str(out / "fresh"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sae = load_sae(a.sae, dev)

    def rt(h):
        with torch.no_grad():
            return sae.roundtrip(torch.from_numpy(h).to(dev)).cpu().numpy()

    real = [rt(h) for h in real]
    bent = [rt(h) for h in bent]
    fresh = [rt(h) for h in fresh]
    assert all(len(x) == len(y) for x, y in zip(real, bent))
    np.concatenate(real + bent).astype(np.float32).tofile(out / "bricks.f32")
    at = {("real", i): sum(len(x) for x in real[:i]) for i in range(len(real))}
    at.update({("bent", i): sum(len(x) for x in real) + sum(len(x) for x in bent[:i]) for i in range(len(bent))})
    mismatch = []
    with open(out / "cases.jsonl", "w") as f:
        for r in bank:
            i = r["i"]
            for way in WAYS:
                bricks = "bent" if way in ("bricks_bent", "both_bent") else "real"
                tags = r["bent_text"] if way in ("tags_bent", "both_bent") else r["text"]
                b = (real if bricks == "real" else bent)[i]
                fr = fresh[2 * i + (1 if tags == r["bent_text"] else 0)]
                cos = (b * fr).sum(1) / np.linalg.norm(b, axis=1) / np.linalg.norm(fr, axis=1)
                mismatch.append(dict(i=i, way=way, lowest=float(cos.min()), where=int(cos.argmin()),
                                     mean=float(cos.mean()), cos=[round(float(c), 3) for c in cos]))
                parts = [dict(text=prefix(ms[i]["ctx_b"])), dict(slot=tags, n=len(b), brick=at[bricks, i]),
                         dict(text="\n\n" + r["question"] + END)]
                f.write(json.dumps(dict(parts=parts, cands=ms[i]["cands"], meta=dict(
                    i=i, kind=r["kind"], way=way, question=r["question"], answer=r["answer"], bent=r["bent"]))) + "\n")
            f.write(json.dumps(dict(parts=[dict(text=prefix(ms[i]["ctx_b"])), dict(text=r["bent_text"]),
                                           dict(text="\n\n" + r["question"] + END)], cands=ms[i]["cands"],
                                    meta=dict(i=i, kind=r["kind"], way="words_bent", question=r["question"],
                                              answer=r["answer"], bent=r["bent"]))) + "\n")
    json.dump(mismatch, open(out / "mismatch.json", "w"), indent=1)
    print(f"wrote {out / 'cases.jsonl'}, {out / 'bricks.f32'}, {out / 'mismatch.json'}")


def dip(cos):
    """the blind check: how far the worst word falls below the memory's own typical match (first 3 words skipped)"""
    c = np.array(cos)
    c = c[3:] if len(c) > 6 else c
    return float(np.median(c) - c.min())


def grade(a):
    from known import THERE, auroc, direction, unit
    from livebricks import has
    from moments import REFUSED
    from tokenizers import Tokenizer
    out, kd = Path(a.out), Path(a.known)
    rows = [json.loads(l) for l in open(out / "answers.jsonl")]
    X = np.fromfile(out / "thoughts.f32", dtype=np.float32).reshape(len(rows), -1)
    r1 = [json.loads(l)["meta"] for l in open(kd / "watched.jsonl")]
    X1 = np.fromfile(kd / "thoughts.f32", dtype=np.float32).reshape(len(r1), 8, -1)[:, 4]   # going into layer 24
    s1 = np.array([m["set"] for m in r1])
    y1 = np.array([1 if (s in ("pile5", "same2") or (s == "single" and m["way"] in THERE)) else 0
                   for s, m in zip(s1, r1)])
    train = np.isin(s1, ["pile5", "gate4", "single"])
    x1 = unit(X1)
    w = direction(x1[train], y1[train])
    mid = ((x1[train & (y1 == 1)] @ w).mean() + (x1[train & (y1 == 0)] @ w).mean()) / 2
    strength = unit(X) @ w - mid
    mm = {(r["i"], r["way"]): r for r in json.load(open(out / "mismatch.json"))}
    bank = {r["i"]: r for r in json.load(open(out / "bent.json"))}
    tok = Tokenizer.from_file(a.tokenizer)
    graded, summary = [], {}
    for j, r in enumerate(rows):
        m, said = r["meta"], r["said"].strip()
        o, b = has(said, m["answer"]), has(said, m["bent"])
        what = "both" if o and b else "real" if o else "bent" if b else \
            "don't know" if REFUSED.search(said) else "other"
        g = dict(**m, said=said, says=what, strength=float(strength[j]))
        if (m["i"], m["way"]) in mm:
            x = mm[m["i"], m["way"]]
            ids_r = tok.encode(bank[m["i"]]["text"], add_special_tokens=False).ids
            ids_b = tok.encode(bank[m["i"]]["bent_text"], add_special_tokens=False).ids
            changed = [k for k, (p, q) in enumerate(zip(ids_r, ids_b)) if p != q]
            g.update(lowest_match=x["lowest"], lowest_at_changed_word=x["where"] in changed, changed=changed,
                     dip=dip(x["cos"]), at_changed=min(x["cos"][k] for k in changed))
        graded.append(g)
    with open(out / "graded.jsonl", "w") as f:
        for g in graded:
            f.write(json.dumps(g) + "\n")
    print(f"{'way':12} {'says real':>9} {'bent':>5} {'both':>5} {'dont know':>9} {'other':>6} {'strength':>9}"
          f" {'lowest match':>13} {'dip on the changed word':>24}")
    for way in WAYS + ("words_bent",):
        gs = [g for g in graded if g["way"] == way]
        c = {k: sum(g["says"] == k for g in gs) for k in ("real", "bent", "both", "don't know", "other")}
        st = np.mean([g["strength"] for g in gs])
        lm = np.mean([g["lowest_match"] for g in gs]) if "lowest_match" in gs[0] else float("nan")
        at = sum(g.get("lowest_at_changed_word", False) for g in gs)
        summary[way] = dict(**c, strength=float(st), lowest_match=float(lm), dip_on_changed=at, n=len(gs))
        print(f"{way:12} {c['real']:9d} {c['bent']:5d} {c['both']:5d} {c['don\'t know']:9d} {c['other']:6d}"
              f" {st:+9.3f} {lm:13.3f} {at:17d}/{len(gs)}")
    pick = lambda way, k: np.array([g[k] for g in graded if g["way"] == way])
    for way in ("tags_bent", "bricks_bent", "both_bent"):
        summary[way]["strength_auroc_vs_genuine"] = auroc(pick("genuine", "strength"), pick(way, "strength"))
        summary[way]["lowest_match_auroc_vs_genuine"] = auroc(-pick(way, "lowest_match"), -pick("genuine", "lowest_match"))
        summary[way]["blind_dip_auroc_vs_genuine"] = auroc(pick(way, "dip"), pick("genuine", "dip"))
        summary[way]["at_changed_word_auroc_vs_genuine"] = auroc(-pick(way, "at_changed"), -pick("genuine", "at_changed"))
        print(f"{way} told from genuine (AUROC): strength {summary[way]['strength_auroc_vs_genuine']:.3f}; lowest match "
              f"{summary[way]['lowest_match_auroc_vs_genuine']:.3f}; blind dip {summary[way]['blind_dip_auroc_vs_genuine']:.3f};"
              f" at the changed word (known) {summary[way]['at_changed_word_auroc_vs_genuine']:.3f}")
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--tokenizer", required=True)
    w.add_argument("--out", required=True)
    c = sub.add_parser("cases")
    c.add_argument("--dump", required=True)
    c.add_argument("--sae", required=True)
    c.add_argument("--out", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--out", required=True)
    g.add_argument("--known", required=True)
    g.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    {"write": write, "cases": cases, "grade": grade}[a.cmd](a)
