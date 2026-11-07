#!/usr/bin/env python3
"""Blind hand audit of the cap-off runs and the science mix. usage: audit_room.py build|lock|score
build: packet in this folder. lock: checks verdicts-v1.jsonl covers every item, records its sha256. score: only after lock, checks every recorded hash, then unblinds."""
import hashlib, json, os, random, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
LAW = os.path.dirname(HERE)
SCI = os.path.join(os.path.dirname(LAW), "science")
sys.path.insert(0, os.path.join(SCI, "grades-v4"))
sys.dont_write_bytecode = True
from keyed_grade import keyed_grade, key_text
ARMS = {
    "reach": {u: f"{LAW}/reach1/reach-{u}.jsonl" for u in ("g", "s1")},
    "f5room": {u: f"{LAW}/f5room1/f5room-{u}.jsonl" for u in ("g", "s1", "s2", "s3")},
    "c5room": {u: f"{LAW}/f5room1/c5room-{u}.jsonl" for u in ("g", "s1")},
    "scimix": {"g": f"{LAW}/mix1/eval-scimix.jsonl", "s1": f"{LAW}/mix1/eval-scimix-s1.jsonl"},
    "scic5": {"g": f"{LAW}/mix1/eval-scic5.jsonl", "s1": f"{LAW}/mix1/eval-scic5-s1.jsonl"},
}
HELD = f"{LAW}/mix1/sci-held-81.jsonl"
TAIL = 1800
jl = lambda p: [json.loads(l) for l in open(p)]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
def die(m):
    print(m); sys.exit(1)
def attempts():
    ids = [r["id"] for r in jl(HELD)]
    out = []
    for arm, files in ARMS.items():
        for u, p in files.items():
            rows = {r["id"]: r for r in jl(p)}
            miss = [i for i in ids if i not in rows]
            if miss:
                die(f"{arm} {u} is missing {len(miss)} of {len(ids)}")
            for i in ids:
                r = rows[i]
                out.append(dict(arm=arm, run=u, id=i, q=r["q"], key=key_text(i, r["answer"], r.get("scale"), r.get("unit")), answer=r["answer"], scale=r.get("scale"), finish=r.get("finish"), reply=r.get("reply") or "", thinking=r.get("thinking") or "", tokens=(r.get("think_tokens") or 0) + (r.get("reply_tokens") or 0)))
    return out
def build():
    if os.path.exists(f"{HERE}/.map-v1.json"):
        die("already built; refusing to reshuffle")
    os.makedirs(f"{HERE}/problems", exist_ok=True)
    A = attempts(); rng = random.SystemRandom()
    ids = sorted({a["id"] for a in A}); rng.shuffle(ids)
    amap, items = {}, []
    with open(f"{HERE}/full-replies.jsonl", "w") as full:
        for n, i in enumerate(ids, 1):
            P = f"P{n:03d}"
            done = [a for a in A if a["id"] == i and a["finish"] == "stop"]
            rng.shuffle(done)
            first = next(a for a in A if a["id"] == i)
            with open(f"{HERE}/problems/{P}.md", "w") as w:
                w.write(f"# {P}\n\n## Question\n\n{first['q']}\n\n## Key\n\n{first['key']}\n\n## Finished replies: {len(done)}\n\n")
                for k, a in enumerate(done):
                    item = f"{P}-{k + 1:02d}"
                    items.append(item)
                    amap[item] = dict(arm=a["arm"], run=a["run"], id=a["id"])
                    rep = a["reply"]
                    w.write(f"### {item}\n\n{'[start not shown, full reply in full-replies.jsonl] ' if len(rep) > TAIL else ''}{rep[-TAIL:] if rep.strip() else '(empty reply)'}\n\n")
                    full.write(json.dumps(dict(item=item, thinking=a["thinking"], reply=rep)) + "\n")
    json.dump(amap, open(f"{HERE}/.map-v1.json", "w"))
    open(f"{HERE}/items.txt", "w").write("\n".join(items) + "\n")
    inp = [HELD] + [p for f in ARMS.values() for p in f.values()]
    json.dump(dict(problems=len(ids), items=len(items), attempts=len(A), not_shown_cut=len(A) - len(items), inputs={p: sha(p) for p in inp}, map_sha256=sha(f"{HERE}/.map-v1.json"), tool_sha256=sha(os.path.abspath(__file__))), open(f"{HERE}/manifest-v1.json", "w"), indent=1)
    print(f"{len(ids)} problems, {len(items)} finished replies to read, {len(A) - len(items)} cut and not shown")
def read_verdicts():
    items = open(f"{HERE}/items.txt").read().split(); V = {}
    for r in jl(f"{HERE}/verdicts-v1.jsonl"):
        if r["item"] in V:
            die(f"duplicate {r['item']}")
        if r["verdict"] not in ("right", "wrong", "unclear"):
            die(f"bad verdict {r}")
        V[r["item"]] = r
    miss, extra = set(items) - set(V), set(V) - set(items)
    if miss or extra:
        die(f"missing {len(miss)} {sorted(miss)[:8]}, extra {len(extra)} {sorted(extra)[:8]}")
    return V
def lock():
    V = read_verdicts()
    open(f"{HERE}/verdicts-v1.sha256", "w").write(sha(f"{HERE}/verdicts-v1.jsonl") + "\n")
    print(f"locked {len(V)} verdicts")
def score():
    lp = f"{HERE}/verdicts-v1.sha256"
    if not os.path.exists(lp):
        die("not locked")
    if open(lp).read().strip() != sha(f"{HERE}/verdicts-v1.jsonl"):
        die("verdicts changed after lock")
    M_ = json.load(open(f"{HERE}/manifest-v1.json"))
    bad = [p for p, h in M_["inputs"].items() if sha(p) != h]
    if bad or sha(f"{HERE}/.map-v1.json") != M_["map_sha256"]:
        die(f"an input or the map changed: {bad}")
    V = read_verdicts(); amap = json.load(open(f"{HERE}/.map-v1.json"))
    A = attempts(); ids = sorted({a["id"] for a in A})
    item_of = {(m["arm"], m["run"], m["id"]): k for k, m in amap.items()}
    M, H, tok = {}, {}, {}
    for a in A:
        k = (a["arm"], a["run"], a["id"])
        g = keyed_grade(a["finish"], a["reply"], a["id"], a["answer"], a["scale"])
        M[k] = "wrong" if g == "sign" else g
        H[k] = V[item_of[k]]["verdict"] if k in item_of else "cut"
        tok[k] = a["tokens"]
    L = []
    def table(G, name):
        for arm, f in ARMS.items():
            for label, runs in ((f"{len(f)} runs", list(f)),) + ((("g+s1", ["g", "s1"]),) if len(f) > 2 else ()):
                c = Counter(G[(arm, u, i)] for u in runs for i in ids); N = len(ids) * len(runs); fin = c["right"] + c["wrong"] + c["unclear"]
                L.append(f"{name:<8} {arm:<7} {label:<7} right {c['right']:>3} ({100*c['right']/N:.1f}%) wrong {c['wrong']:>3} ({100*c['wrong']/N:.1f}%) unclear {c['unclear']:>2} cut {c['cut']:>3} ({100*c['cut']/N:.1f}%) | right among finished {100*c['right']/(fin or 1):.1f}% | tokens {sum(tok[(arm, u, i)] for u in runs for i in ids)}")
        L.append("")
    table(M, "MACHINE"); table(H, "HAND")
    N = len(ids) * 2; cr = Counter(H[("reach", u, i)] for u in ("g", "s1") for i in ids)
    hard = set(json.load(open(f"{LAW}/reach1/f5-wrong-twice-ids-v1.json")))
    ph = Counter(H[("reach", u, i)] for u in ("g", "s1") for i in hard)
    r, cu = cr["right"] / N, cr["cut"] / N
    L.append(f"reach bars by hand: {'STILL CUTTING' if cu > .25 else 'THE TARGET IS IN REACH' if r >= .8 else 'MORE IS IN REACH, NOT ALL OF IT' if r >= .69 else 'SAME CEILING AS F5'} (R {100*r:.1f}%, cut {100*cu:.1f}%); wrong-twice P {100*ph['right']/26:.1f}%, {'ROOM FIXES F5S WRONGS' if ph['right']/26 >= .5 else 'OUT OF REACH' if ph['right']/26 <= .2 else 'PARTLY'}")
    F = Counter(H[("f5room", u, i)] for u in ARMS["f5room"] for i in ids); n4 = len(ids) * 4
    L.append(f"f5room bars by hand: R {100*F['right']/n4:.1f}% {'ROOM ALONE REACHES THE TARGET' if F['right']/n4 >= .8 else 'ROOM GETS HIM MOST OF THE WAY' if F['right']/n4 >= .72 else 'ROOM IS NOT ENOUGH'}; W {100*F['wrong']/n4:.1f}% {'THE WRONGS HOLD' if F['wrong']/n4 <= .18 else 'THE WRONGS GROW' if F['wrong']/n4 > .22 else 'SOME NEW WRONGS'}")
    open(f"{HERE}/RESULT-v1.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    with open(f"{HERE}/disagreements-v1.txt", "w") as w:
        for k, m in sorted(amap.items()):
            t = (m["arm"], m["run"], m["id"])
            if M[t] != H[t]:
                w.write(f"{k} {m['arm']} {m['run']} {m['id']} machine {M[t]} hand {H[t]} | {V[k].get('note', '')}\n")
    for arm, f in ARMS.items():
        json.dump({f"{u}:{i}": H[(arm, u, i)] for u in f for i in ids if H[(arm, u, i)] in ("right", "wrong") and H[(arm, u, i)] != M[(arm, u, i)]}, open(f"{HERE}/corrections-{arm}-v1.json", "w"), indent=0)
if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("build", "lock", "score"):
        die(__doc__)
    dict(build=build, lock=lock, score=score)[sys.argv[1]]()
