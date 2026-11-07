#!/usr/bin/env python3
"""Blind hand audit: build packets, lock verdicts, then unblind and score.
usage: audit_tool.py build|lock|score song|flip
build: writes <audit>/problems/*.md, items.txt, full-replies.jsonl, manifest-v1.json and the hidden .map-v1.json.
lock:  checks <audit>/verdicts-v1.jsonl covers every item, records its sha256.
score: only after lock. Reads the map and writes RESULT-v1.txt, disagreements-v1.txt, corrections files.
AUDIT_ROOT and AUDIT_SONG_DIR move the output and the new rows, for testing."""
import hashlib, json, os, random, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIT_ROOT", HERE)
LFM = os.path.abspath(os.path.join(HERE, "..", ".."))
SCI = os.path.join(LFM, "science")
sys.path.insert(0, os.path.join(SCI, "grades-v4"))
from keyed_grade import keyed_grade, key_text
MIX = os.environ.get("AUDIT_SONG_DIR", os.path.join(LFM, "law-1105", "mix1"))
FLIP = os.path.join(LFM, "law-1105", "flip1")
RUNS = ("g", "s1", "s2", "s3")
CLOSE = "<channel|>"
TAIL = 1800
VERDICTS = ("right", "wrong", "unclear")
def jl(p):
    return [json.loads(l) for l in open(p)]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
def die(m):
    print(m); sys.exit(1)
def song_inputs():
    a = {u: f"{SCI}/final-f5-1104/f5-{u}.jsonl" for u in RUNS}
    b = {u: f"{MIX}/f5song-{u}.jsonl" for u in RUNS}
    return a, b, f"{MIX}/sci-held-81.jsonl"
def song_attempts():
    a, b, h = song_inputs()
    for u in RUNS:
        if not os.path.exists(b[u]):
            die(f"not ready: {b[u]} is missing")
    ids = [r["id"] for r in jl(h)]
    out = []
    for arm, files in (("f5", a), ("song", b)):
        for u in RUNS:
            rows = {r["id"]: r for r in jl(files[u])}
            miss = [i for i in ids if i not in rows]
            if miss:
                die(f"not ready: {arm} {u} is missing {len(miss)} of {len(ids)} problems")
            for i in ids:
                r = rows[i]
                out.append(dict(arm=arm, run=u, id=i, q=r["q"], key=key_text(i, r["answer"], r.get("scale"), r.get("unit")), answer=r["answer"], scale=r.get("scale"), finish=r.get("finish"), reply=r.get("reply") or "", thinking=r.get("thinking") or "", tokens=(r.get("think_tokens") or 0) + (r.get("reply_tokens") or 0)))
    return out
def flip_attempts():
    steps = {r["id"]: r for r in jl(f"{FLIP}/flip-steps-v1.jsonl")}
    out = []
    for r in jl(f"{FLIP}/e4b-flip-srv-v1.jsonl"):
        t = r["text"] or ""
        th, rep = t.split(CLOSE, 1) if CLOSE in t else (t, "")
        fin = "stop" if r["done"] and rep else "length"
        out.append(dict(arm=r["arm"], run=r["set"], id=r["id"], q=steps[r["id"]]["q"], key=key_text(r["id"], r["answer"], r.get("scale"), None), answer=r["answer"], scale=r.get("scale"), finish=fin, reply=rep, thinking=th, tokens=r.get("tokens") or 0))
    return out
def attempts(audit):
    return song_attempts() if audit == "song" else flip_attempts()
def build(audit):
    d = os.path.join(ROOT, audit)
    if os.path.exists(os.path.join(d, ".map-v1.json")):
        die(f"{d} already has a map; refusing to shuffle a packet that may be in use")
    os.makedirs(os.path.join(d, "problems"), exist_ok=True)
    A = attempts(audit)
    rng = random.SystemRandom()
    ids = sorted({a["id"] for a in A})
    rng.shuffle(ids)
    amap, items = {}, []
    with open(os.path.join(d, "full-replies.jsonl"), "w") as full:
        for n, i in enumerate(ids, 1):
            P = f"P{n:03d}"
            done = [a for a in A if a["id"] == i and a["finish"] == "stop"]
            rng.shuffle(done)
            first = next(a for a in A if a["id"] == i)
            with open(os.path.join(d, "problems", f"{P}.md"), "w") as w:
                w.write(f"# {P}\n\n## Question\n\n{first['q']}\n\n## Key\n\n{first['key']}\n\n## Finished replies: {len(done)}\n\n")
                for k, a in enumerate(done):
                    item = f"{P}-{chr(65 + k)}"
                    items.append(item)
                    amap[item] = dict(arm=a["arm"], run=a["run"], id=a["id"])
                    rep = a["reply"]
                    cut = len(rep) > TAIL
                    w.write(f"### {item}\n\n{'[start not shown, full reply in full-replies.jsonl] ' if cut else ''}{rep[-TAIL:] if rep.strip() else '(empty reply)'}\n\n")
                    full.write(json.dumps(dict(item=item, thinking=a["thinking"], reply=rep)) + "\n")
    with open(os.path.join(d, ".map-v1.json"), "w") as w:
        json.dump(amap, w)
    with open(os.path.join(d, "items.txt"), "w") as w:
        w.write("\n".join(items) + "\n")
    if audit == "song":
        a, b, h = song_inputs()
        inp = [*a.values(), *b.values(), h]
    else:
        inp = [f"{FLIP}/flip-steps-v1.jsonl", f"{FLIP}/e4b-flip-srv-v1.jsonl"]
    json.dump(dict(audit=audit, problems=len(ids), items=len(items), attempts=len(A), not_shown_cut=len(A) - len(items), inputs={p: sha(p) for p in inp}, map_sha256=sha(os.path.join(d, ".map-v1.json"))), open(os.path.join(d, "manifest-v1.json"), "w"), indent=1)
    print(f"{audit}: {len(ids)} problems, {len(items)} finished replies to read, {len(A) - len(items)} cut and not shown")
def read_verdicts(d):
    items = open(os.path.join(d, "items.txt")).read().split()
    V = {}
    for r in jl(os.path.join(d, "verdicts-v1.jsonl")):
        if r["item"] in V:
            die(f"duplicate verdict {r['item']}")
        if r["verdict"] not in VERDICTS:
            die(f"bad verdict {r}")
        V[r["item"]] = r
    miss, extra = set(items) - set(V), set(V) - set(items)
    if miss or extra:
        die(f"missing {len(miss)} {sorted(miss)[:8]}, extra {len(extra)} {sorted(extra)[:8]}")
    return V
def lock(audit):
    d = os.path.join(ROOT, audit)
    V = read_verdicts(d)
    open(os.path.join(d, "verdicts-v1.sha256"), "w").write(sha(os.path.join(d, "verdicts-v1.jsonl")) + "\n")
    print(f"locked {len(V)} verdicts")
def bar(dr, dw, step=16):
    if dr <= -step or dw >= step:
        return "IT HURTS"
    if dr >= step and dw <= -step:
        return "BOTH MOVE THE RIGHT WAY"
    if dr >= step:
        return "MORE RIGHT"
    if dw <= -step:
        return "FEWER WRONG"
    return "NOTHING"
def machine(a):
    g = keyed_grade(a["finish"], a["reply"], a["id"], a["answer"], a["scale"])
    return "wrong" if g == "sign" else g
def score_song(d, V, amap, say):
    import numpy as np
    A = song_attempts()
    item_of = {(m["arm"], m["run"], m["id"]): k for k, m in amap.items()}
    M, H = {}, {}
    for a in A:
        k = (a["arm"], a["run"], a["id"])
        M[k] = machine(a)
        H[k] = V[item_of[k]]["verdict"] if k in item_of else "cut"
    ids = sorted({a["id"] for a in A})
    N = len(ids) * 4
    tok = {arm: sum(a["tokens"] for a in A if a["arm"] == arm) for arm in ("f5", "song")}
    def table(G, arm, label):
        c = {x: sum(G[(arm, u, i)] == x for u in RUNS for i in ids) for x in ("right", "wrong", "unclear", "cut")}
        fin = c["right"] + c["wrong"] + c["unclear"]
        once = sum(any(G[(arm, u, i)] == "right" for u in RUNS) for i in ids)
        by = [sum(G[(arm, u, i)] == "right" for i in ids) for u in RUNS]
        say(f"{label:<14} {arm:<5} right {c['right']} ({100*c['right']/N:.1f}%) wrong {c['wrong']} ({100*c['wrong']/N:.1f}%) unclear {c['unclear']} cut {c['cut']} ({100*c['cut']/N:.1f}%) | right among finished {100*c['right']/max(fin,1):.1f}% | right at least once in 4: {once} of {len(ids)} ({100*once/len(ids):.1f}%) | right by run {by}")
        return c
    rng = np.random.default_rng(1105)
    def ci(G, kind):
        x = np.array([sum(G[("song", u, i)] == kind for u in RUNS) - sum(G[("f5", u, i)] == kind for u in RUNS) for i in ids])
        v = [x[rng.integers(0, len(x), len(x))].sum() for _ in range(4000)]
        return f"{int(x.sum()):+d} ({np.percentile(v, 2.5):+.0f} to {np.percentile(v, 97.5):+.0f})"
    say(f"{len(ids)} problems, 4 runs each, {N} runs per arm. Both arms use the same standard.\n")
    for name, G in (("MACHINE", M), ("HAND", H)):
        a = table(G, "f5", name)
        b = table(G, "song", name)
        dr, dw = b["right"] - a["right"], b["wrong"] - a["wrong"]
        say(f"{name} change, song minus f5, 95% interval by problem: right {ci(G, 'right')}, wrong {ci(G, 'wrong')}, cut {ci(G, 'cut')}")
        say(f"{name} VERDICT{' (official, as written in the goal)' if name == 'MACHINE' else ' (same thresholds, reported beside it)'}: {bar(dr, dw)}")
        say(f"{name} target: right 80% {'reached' if b['right']/N >= .8 else 'not reached'}, wrong under 10% {'reached' if b['wrong']/N < .1 else 'not reached'}\n")
    say(f"tokens f5 {tok['f5']} song {tok['song']} change {100*(tok['song']/tok['f5']-1):+.1f}%\n")
    say("HAND moves, f5 grade to song grade, same run label and problem:")
    for x in ("right", "wrong", "unclear", "cut"):
        say("  " + x + " to " + ", ".join(f"{y} {sum(H[('f5', u, i)] == x and H[('song', u, i)] == y for u in RUNS for i in ids)}" for y in ("right", "wrong", "unclear", "cut")))
    with open(os.path.join(d, "disagreements-v1.txt"), "w") as w:
        for k, m in sorted(amap.items()):
            t = (m["arm"], m["run"], m["id"])
            if M[t] != H[t]:
                w.write(f"{k} {m['arm']} {m['run']} {m['id']} machine {M[t]} hand {H[t]} | {V[k].get('note', '')}\n")
    for arm in ("f5", "song"):
        json.dump({f"{u}:{i}": H[(arm, u, i)] for u in RUNS for i in ids if H[(arm, u, i)] in ("right", "wrong") and H[(arm, u, i)] != M[(arm, u, i)]}, open(os.path.join(d, f"corrections-{arm}-v1.json"), "w"), indent=0)
def score_flip(d, V, amap, say):
    fix = {f"{m['id']}:{m['arm']}": V[k]["verdict"] for k, m in amap.items()}
    p = os.path.join(d, "corrections-flip-v1.json")
    json.dump(fix, open(p, "w"), indent=0)
    py, script, rows = sys.executable, f"{FLIP}/flip_score.py", f"{FLIP}/e4b-flip-srv-v1.jsonl"
    say("MACHINE, official standard in the goal is the hand read, this is for comparison\n" + subprocess.run([py, script, rows], capture_output=True, text=True, check=True).stdout)
    say("HAND, every finished reply read blind\n" + subprocess.run([py, script, rows, p], capture_output=True, text=True, check=True).stdout)
    by = {(a["id"], a["arm"]): a for a in flip_attempts()}
    with open(os.path.join(d, "disagreements-v1.txt"), "w") as w:
        for k, m in sorted(amap.items()):
            g = machine(by[(m["id"], m["arm"])])
            if g != V[k]["verdict"]:
                w.write(f"{k} {m['arm']} set {m['run']} {m['id']} machine {g} hand {V[k]['verdict']} | {V[k].get('note', '')}\n")
def score(audit):
    d = os.path.join(ROOT, audit)
    lp = os.path.join(d, "verdicts-v1.sha256")
    if not os.path.exists(lp):
        die("not locked: run lock first")
    if open(lp).read().strip() != sha(os.path.join(d, "verdicts-v1.jsonl")):
        die("verdicts changed after lock")
    V = read_verdicts(d)
    amap = json.load(open(os.path.join(d, ".map-v1.json")))
    lines = []
    def say(s):
        print(s); lines.append(s)
    (score_song if audit == "song" else score_flip)(d, V, amap, say)
    open(os.path.join(d, "RESULT-v1.txt"), "w").write("\n".join(lines) + "\n")
if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "lock", "score") or sys.argv[2] not in ("song", "flip"):
        die(__doc__)
    dict(build=build, lock=lock, score=score)[sys.argv[1]](sys.argv[2])
