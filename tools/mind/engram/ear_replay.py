"""His ear for people, replayed: how fast does he learn who writes through each way in, from nothing?

Immy, 2026-10-22: "just like a baby learns its parents' voice ... he should learn our style first". His live mind
(tools/room/mind_live.py, hear_voice) keeps a running print per way in (brainstem /print: his state going into layer
12 on the message's last word) and, once two ways are each known from VOICE_MIN messages, says a new message sounds
like the print it points toward (each print against the mean of everything heard). This replays that, with the same
math, on real messages from whois.py's pool, held out by conversation like whois.py: the test conversations (a third
of "main"'s, three folds in turn) are never heard; a stream of Immy's and Claude's messages from the other
conversations is heard one at a time in a random order (Immy's through "menubar", Claude's through "terminal"), and at
each checkpoint the ear (frozen there) is tried on the fold's test messages and on their real chats with him
("to_him", 8 + 8, whose conversations are never heard either). Folds x random orders give the spread.

  python ear_replay.py --pool POOL.json --key KEY --port 8781 --out DIR [--per 300] [--orders 5]
  -> DIR/prints.npz (every print, cached), DIR/curve.json, DIR/curve.txt
"""
import argparse, json, random, urllib.request
from pathlib import Path

import numpy as np

VOICE_MIN = 10                                   # the same as mind_live.py
CHECK = [10, 20, 40, 80, 160, 300]               # messages heard from each person when the ear is tried


def call(args, path, **req):
    r = urllib.request.Request(f"http://127.0.0.1:{args.port}/{path}", data=json.dumps(req).encode(),
                               headers={"X-LG-Key": args.key_text, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.loads(f.read())


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def judge(sum_by, n_by, sum_all, n_all, X):
    """the ear frozen as it is: for each print, its score toward each known way in (mind_live.sounds_like)"""
    c = sum_all / n_all
    d = X - c
    s = {}
    for w in ("menubar", "terminal"):
        p = sum_by[w] / n_by[w] - c
        s[w] = d @ p / (np.linalg.norm(d, axis=1) * np.linalg.norm(p) + 1e-12)
    return s


def run(a):
    a.key_text = open(a.key).read().strip()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(a.pool))
    sessions = sorted({x["session"] for x in pool["main"]})
    him = {x["session"] for x in pool["to_him"]}
    folds = []
    for f in range(3):
        held = set(sessions[f::3])
        stream = {w: [x for x in pool["others"] if x["who"] == w and x["session"] not in held | him]
                  for w in ("immy", "claude")}
        rng = random.Random(f)
        for w in stream:
            rng.shuffle(stream[w])
            stream[w] = stream[w][:a.per]
        folds.append((held, stream))
    texts = ([x["text"] for x in pool["main"]] + [x["text"] for x in pool["to_him"]]
             + [x["text"] for _, st in folds for w in st for x in st[w]])
    cache = out / "prints.npz"
    if cache.exists():
        P = dict(np.load(cache, allow_pickle=False))
        have = {t: P["X"][i] for i, t in enumerate(P["texts"])}
    else:
        have = {}
    for i, t in enumerate(texts):
        if t not in have:
            have[t] = np.array(call(a, "print", text=t[:8000])["print"], dtype=np.float32)
            if len(have) % 50 == 0:
                print("prints", len(have), "of", len(set(texts)), flush=True)
    keys = list(have)
    np.savez(cache, texts=np.array(keys), X=np.stack([have[k] for k in keys]))
    way = {"immy": "menubar", "claude": "terminal"}
    curve = {name: {k: [] for k in CHECK} for name in ("main", "to_him")}
    for (held, stream), o in [(fs, o) for fs in folds for o in range(a.orders)]:
        tests = {}
        for name in ("main", "to_him"):
            xs = [x for x in pool[name] if name == "to_him" or x["session"] in held]
            tests[name] = (np.stack([have[x["text"]] for x in xs]), [way[x["who"]] for x in xs])
        heard = [(way[w], have[x["text"]]) for w in stream for x in stream[w]]
        random.Random(100 + o).shuffle(heard)
        sum_by = {w: np.zeros_like(heard[0][1]) for w in way.values()}
        n_by = {w: 0 for w in way.values()}
        sum_all, n_all = np.zeros_like(heard[0][1]), 0
        todo = list(CHECK)
        for w, x in heard:
            sum_by[w] += x; n_by[w] += 1; sum_all += x; n_all += 1
            while todo and min(n_by.values()) >= todo[0]:
                k = todo.pop(0)
                for name, (X, truth) in tests.items():
                    s = judge(sum_by, n_by, sum_all, n_all, X)
                    like = np.where(s["menubar"] >= s["terminal"], "menubar", "terminal")
                    margin = s["menubar"] - s["terminal"]                # toward Immy's way in
                    im = np.array([t == "menubar" for t in truth])
                    curve[name][k].append(dict(
                        right=int((like == np.array(truth)).sum()), of=len(truth),
                        immy_right=int((like[im] == "menubar").sum()), claude_right=int((like[~im] == "terminal").sum()),
                        auroc=auroc(margin[im], margin[~im])))
    json.dump(curve, open(out / "curve.json", "w"), indent=1)
    lines = [f"stream: up to {a.per} of Immy's + {a.per} of Claude's messages from conversations outside the tests, "
             f"3 folds x {a.orders} random orders; the ear is tried when it has heard k from each"]
    for name in curve:
        lines.append(f"\n{name} (per fold and order; counts are means)")
        for k in CHECK:
            rs = curve[name][k]
            if not rs:
                continue
            acc = [r["right"] / r["of"] for r in rs]
            of = np.mean([r["of"] for r in rs])
            au = [r["auroc"] for r in rs]
            lines.append(f"  after {k:>3} each: right {np.mean(acc):.1%} (+-{np.std(acc):.1%}) of {of:.0f}, Immy's "
                         f"{np.mean([r['immy_right'] for r in rs]):.0f}, Claude's {np.mean([r['claude_right'] for r in rs]):.0f}, "
                         f"AUROC {np.mean(au):.3f} (+-{np.std(au):.3f})")
    txt = "\n".join(lines)
    print(txt)
    (out / "curve.txt").write_text(txt + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True); p.add_argument("--key", required=True)
    p.add_argument("--port", type=int, default=8781); p.add_argument("--out", required=True)
    p.add_argument("--per", type=int, default=300); p.add_argument("--orders", type=int, default=5)
    run(p.parse_args())
