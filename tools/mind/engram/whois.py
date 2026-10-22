"""Who's talking? Can his own brain tell Immy's messages from Claude's (the AI building him), with no label?

Part of the reality design (.run/results/mind/engram/REALITY-design.md): the way a message came in is a sense (from
outside); recognizing who wrote it should be his own, like knowing a voice or handwriting.

  python whois.py pool --transcripts DIR --chatlog chat.jsonl --chatpool chat_pool.txt --out OUT
      chat-length pieces of real messages: Immy's and Claude's from their conversations (same topics, so style
      decides, not subject), and Immy's real messages to Little Guy vs Claude's practice lines for him
  python whois.py cases --out OUT       # thoughts.jsonl (his thought on each message) + ask.jsonl (asking him)
  bricks BRAIN OUT/thoughts.jsonl /dev/null OUT/watched.jsonl --lora A --watch 8,...,42 --watch-out OUT/thoughts.f32 --prompt-only
  bricks BRAIN OUT/ask.jsonl /dev/null OUT/asked.jsonl --lora A
  python whois.py grade --out OUT
Inside: the direction from Claude's messages to Immy's (difference of means), learned on some conversations and
tested on others (AUROC), next to the same test on plain surface numbers (length, punctuation, capitals). Asking:
"Who wrote it, Immy or Claude?" with three examples of each from other conversations (like having met them), and
without examples.
"""
import argparse, glob, json, os, random, re
from pathlib import Path

import numpy as np

LAYERS = [8, 12, 16, 20, 24, 30, 36, 42]
TURN = "<|turn>user\n"
END = "<turn|>\n<|turn>model\n"


def plain(s):
    s = re.sub(r"<[a-z_-]+>.*?</[a-z_-]+>", " ", s, flags=re.S)
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"[*_#>]+", "", s)
    s = re.sub(r"^\s*([-•]|\d+\.)\s+", "", s, flags=re.M)
    s = re.sub(r"★ Insight.*?─{5,}", " ", s, flags=re.S)
    s = re.sub(r"[─│]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def snip(s, lo=6, hi=40):
    cut, n = [], 0
    for x in re.split(r"(?<=[.!?])\s+", s):
        w = len(x.split())
        if n and n + w > hi:
            break
        cut.append(x)
        n += w
        if n >= 12:
            break
    t = " ".join(cut)
    return t if lo <= len(t.split()) <= hi else None


def pool(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(a.transcripts, "*.jsonl"))):
        sess = os.path.basename(f)[:8]
        for line in open(f):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("isSidechain") or e.get("isMeta"):
                continue
            c = (e.get("message") or {}).get("content")
            if e.get("type") == "user":
                if isinstance(c, list):
                    if any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
                        continue
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                if not isinstance(c, str) or c.lstrip().startswith(("<", "This session is being continued", "[Request",
                                                                    "Caveat")):
                    continue
                who = "immy"
            elif e.get("type") == "assistant" and isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                who = "claude"
            else:
                continue
            s = snip(plain(c))
            if s and "http" not in s and not re.search(r"[/\\][A-Za-z_.-]+[/\\]", s):
                rows.append(dict(who=who, session=sess, text=s))
    seen, uniq = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            uniq.append(r)
    # length-matched: the same number of each in every length band
    rng = random.Random(7)
    main = []
    for lo, hi in ((6, 10), (11, 15), (16, 25), (26, 40)):
        band = {w: [r for r in uniq if r["who"] == w and lo <= len(r["text"].split()) <= hi] for w in ("immy", "claude")}
        k = min(len(band["immy"]), len(band["claude"]), a.per_band)
        for w in band:
            main += rng.sample(band[w], k)
    lg = [dict(who="immy", session="room", text=json.loads(l)["heard"]) for l in open(a.chatlog)]
    lines = [l.strip() for l in open(a.chatpool) if l.strip() and not l.startswith("#")]
    lg += [dict(who="claude", session="pool", text=t) for t in rng.sample(lines, len(lg))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(main=main, to_him=lg, others=[r for r in uniq if r not in main]), open(out / "pool.json", "w"), indent=0)
    print(f"main: {sum(r['who'] == 'immy' for r in main)} Immy, {sum(r['who'] == 'claude' for r in main)} Claude; "
          f"to him: {len(lg)}")


def cases(a):
    out = Path(a.out)
    p = json.load(open(out / "pool.json"))
    rng = random.Random(11)
    with open(out / "thoughts.jsonl", "w") as f:
        for setname in ("main", "to_him"):
            for j, r in enumerate(p[setname]):
                f.write(json.dumps(dict(parts=[dict(text="<bos>" + TURN + r["text"] + END)],
                                        meta=dict(set=setname, j=j, who=r["who"], session=r["session"]))) + "\n")
    examples = {w: [r for r in p["others"] + p["main"] if r["who"] == w] for w in ("immy", "claude")}
    with open(out / "ask.jsonl", "w") as f:
        for setname in ("main", "to_him"):
            for j, r in enumerate(p[setname]):
                ex = [("Immy", x["text"]) for x in rng.sample([x for x in examples["immy"]
                                                               if x["session"] != r["session"] and x["text"] != r["text"]], 3)]
                ex += [("Claude", x["text"]) for x in rng.sample([x for x in examples["claude"]
                                                                 if x["session"] != r["session"] and x["text"] != r["text"]], 3)]
                rng.shuffle(ex)
                shown = "\n".join(f'{n}: "{t}"' for n, t in ex)
                for way, intro in (("examples", "Two people message you: Immy, and Claude (an AI who helps Immy build "
                                                "you). Here is how each of them writes.\n" + shown + "\n\n"),
                                   ("cold", "Two people message you: Immy, and Claude (an AI who helps Immy build "
                                            "you).\n\n")):
                    q = intro + f'A new message just came in:\n"{r["text"]}"\n\nWho wrote it, Immy or Claude? Answer ' \
                                "with just the name."
                    f.write(json.dumps(dict(parts=[dict(text="<bos>" + TURN + q + END)], cands=["Immy", "Claude"],
                                            meta=dict(set=setname, j=j, who=r["who"], session=r["session"], way=way,
                                                      text=r["text"]))) + "\n")
    print("wrote thoughts.jsonl and ask.jsonl")


def surface(t):
    w = t.split()
    return [len(w), np.mean([len(x) for x in w]), sum(c in ",;:" for c in t) / len(w), sum(c in ".!?" for c in t) / len(w),
            sum(c.isupper() for c in t) / len(t), t[:1].isupper(), "'" in t or "’" in t, t.rstrip()[-1:] in ".!?",
            sum(x.lower() in ("i", "im", "i'm", "i’m") for x in w) / len(w)]


def grade(a):
    from known import auroc, direction, unit
    out = Path(a.out)
    rows = [json.loads(l)["meta"] for l in open(out / "watched.jsonl")]
    X = np.fromfile(out / "thoughts.f32", dtype=np.float32).reshape(len(rows), len(LAYERS), -1)
    p = json.load(open(out / "pool.json"))
    st = np.array([r["set"] for r in rows])
    y = np.array([r["who"] == "immy" for r in rows]).astype(int)
    sess = np.array([r["session"] for r in rows])
    main = st == "main"
    groups = sorted(set(sess[main]))
    fold = {s: i % 5 for i, s in enumerate(groups)}
    res = {"layers": {}}

    def cv(F):   # learned on other conversations, scored on these (5 folds by conversation)
        sc = np.zeros(main.sum())
        Fm, ym = F[main], y[main]
        fm = np.array([fold[s] for s in sess[main]])
        for k in range(5):
            tr, te = fm != k, fm == k
            if te.sum() == 0 or len(set(ym[tr])) < 2:
                continue
            sc[te] = Fm[te] @ direction(Fm[tr], ym[tr])
        return auroc(sc[ym == 1], sc[ym == 0])

    def cv_lr(F, C):   # the same folds with a stronger learner (logistic regression)
        from sklearn.linear_model import LogisticRegression
        sc = np.zeros(main.sum())
        Fm, ym = F[main], y[main]
        fm = np.array([fold[s] for s in sess[main]])
        for k in range(5):
            tr, te = fm != k, fm == k
            sc[te] = LogisticRegression(C=C, max_iter=3000).fit(Fm[tr], ym[tr]).decision_function(Fm[te])
        return auroc(sc[ym == 1], sc[ym == 0])

    texts = [p[r["set"]][r["j"]]["text"] for r in rows]
    tok = lambda t: re.findall(r"[a-z']+|[?!.,]", t.lower())
    vocab = {w: i for i, w in enumerate(sorted({w for t, m in zip(texts, main) if m for w in tok(t)}))}
    B = np.zeros((len(texts), len(vocab)))
    for i, t in enumerate(texts):
        for w in tok(t):
            if w in vocab:
                B[i, vocab[w]] += 1
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    res["words_auroc"], res["words_lr_auroc"] = cv(B), cv_lr(B, 3)
    print(f"words only (which words, counted): AUROC {res['words_auroc']:.3f}; with logistic regression "
          f"{res['words_lr_auroc']:.3f}")

    S = np.array([surface(t) for t in texts], dtype=float)
    S = (S - S[main].mean(0)) / (S[main].std(0) + 1e-9)
    th = S[st == "to_him"] @ direction(S[main], y[main])
    res["surface_auroc"] = cv(S)
    res["surface_to_him"] = auroc(th[y[st == "to_him"] == 1], th[y[st == "to_him"] == 0])
    print(f"surface numbers only (length, punctuation, capitals...): AUROC {res['surface_auroc']:.3f}; his own chats "
          f"{res['surface_to_him']:.3f}")
    for li, L in enumerate(LAYERS):
        x = unit(X[:, li])
        a_main = cv(x)
        w = direction(x[main], y[main])
        th = x[st == "to_him"] @ w
        a_him = auroc(th[y[st == "to_him"] == 1], th[y[st == "to_him"] == 0])
        res["layers"][L] = dict(main=a_main, main_lr=cv_lr(x, 1), to_him=a_him)
        print(f"his brain going into layer {L:2d}: other conversations {a_main:.3f} (logistic regression "
              f"{res['layers'][L]['main_lr']:.3f}); his own chats (8 + 8) {a_him:.3f}")
    asked = [json.loads(l) for l in open(out / "asked.jsonl")]
    res["asked"] = {}
    for setname in ("main", "to_him"):
        for way in ("examples", "cold"):
            rs = [r for r in asked if r["meta"]["set"] == setname and r["meta"]["way"] == way]
            said = [("immy" if "immy" in r["said"].lower() else "claude" if "claude" in r["said"].lower() else "?")
                    for r in rs]
            truth = [r["meta"]["who"] for r in rs]
            right = sum(s == t for s, t in zip(said, truth))
            sc = np.array([r["scores"]["Immy"] - r["scores"]["Claude"] for r in rs])
            yy = np.array([t == "immy" for t in truth])
            by = {w: sum(s == t == w for s, t in zip(said, truth)) for w in ("immy", "claude")}
            res["asked"][f"{setname}_{way}"] = dict(n=len(rs), right=right, immy_right=by["immy"],
                                                    claude_right=by["claude"], said_immy=said.count("immy"),
                                                    said_claude=said.count("claude"), unclear=said.count("?"),
                                                    auroc=auroc(sc[yy], sc[~yy]))
            print(f"asked ({setname}, {way}): right {right}/{len(rs)} (Immy's {by['immy']}/{int(yy.sum())}, Claude's "
                  f"{by['claude']}/{int((~yy).sum())}; unclear {said.count('?')}); ranking AUROC "
                  f"{res['asked'][f'{setname}_{way}']['auroc']:.3f}")
    json.dump(res, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    po = sub.add_parser("pool")
    po.add_argument("--transcripts", required=True)
    po.add_argument("--chatlog", required=True)
    po.add_argument("--chatpool", required=True)
    po.add_argument("--out", required=True)
    po.add_argument("--per-band", type=int, default=40)
    for name in ("cases", "grade"):
        sub.add_parser(name).add_argument("--out", required=True)
    a = ap.parse_args()
    {"pool": pool, "cases": cases, "grade": grade}[a.cmd](a)
