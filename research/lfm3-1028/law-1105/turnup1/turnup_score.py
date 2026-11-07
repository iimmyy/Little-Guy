#!/usr/bin/env python3
"""GOAL-turnup-v1. usage: turnup_score.py scan <rows> | held <rows> | c5 <plain rows> <c5 rows>"""
import json, sys
import numpy as np
cmd = sys.argv[1]
load = lambda f: [json.loads(l) for l in open(f)]
KC = lambda p: ((p[0] + p[1]) / 2, (p[0] - p[1]) / 2)
if cmd == "scan":
    rows = load(sys.argv[2]); keys = list(rows[0]["reads"])
    print("steps", len(rows), "problems", len({r["id"] for r in rows}))
    t = []
    for k in keys:
        dk = [KC(r["reads"][k])[0] - KC(r["plain"])[0] for r in rows]
        dc = [KC(r["reads"][k])[1] - KC(r["plain"])[1] for r in rows]
        t.append((float(np.median(dk)), float(np.median(dc)), k))
    for w in ("yes", "wait"):
        for mode in ("last", "all"):
            print(w, mode)
            for dk, dc, k in t:
                if k.startswith(f"{w}:{mode}:"):
                    print(f"  L{k.split(':')[2]:>2}  dK {dk:+.2f}  dC {dc:+.2f}")
    ok = sorted([x for x in t if x[1] <= .25], reverse=True)[:6]
    print("largest K rise with C not rising more than 0.25:")
    for dk, dc, k in ok:
        print(f"  {k}  dK {dk:+.2f}  dC {dc:+.2f}")
    sys.exit()
def boot(x, ids, rng=np.random.default_rng(1105), n=2000):
    u = np.unique(ids); g = {p: np.where(ids == p)[0] for p in u}
    v = [np.median(x[np.concatenate([g[p] for p in rng.choice(u, len(u))])]) for _ in range(n)]
    return float(np.median(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
if cmd == "held":
    rows = load(sys.argv[2]); ids = np.array([r["id"] for r in rows])
    print("steps", len(rows), "problems", len(set(ids)))
    K0 = np.array([KC(r["reads"]["plain"])[0] for r in rows]); C0 = np.array([KC(r["reads"]["plain"])[1] for r in rows])
    print(f"plain K {np.median(K0):+.2f}  C {np.median(C0):+.2f}")
    R = {}
    for k in rows[0]["reads"]:
        if k == "plain":
            continue
        dk = np.array([KC(r["reads"][k])[0] for r in rows]) - K0; dc = np.array([KC(r["reads"][k])[1] for r in rows]) - C0
        R[k] = (boot(dk, ids), boot(dc, ids))
        mr = np.median([r["reads"][k][0] - r["reads"]["plain"][0] for r in rows]); mw = np.median([r["reads"][k][1] - r["reads"]["plain"][1] for r in rows])
        print(f"{k:<11} dK {R[k][0][0]:+.2f} ({R[k][0][1]:+.2f} to {R[k][0][2]:+.2f})  dC {R[k][1][0]:+.2f} ({R[k][1][1]:+.2f} to {R[k][1][2]:+.2f})  right page {mr:+.2f}  wrong page {mw:+.2f}")
    z = max(abs(a - b) for r in rows for a, b in zip(r["reads"]["knob:0"], r["reads"]["plain"]))
    print(f"check, alpha 0 vs plain: largest difference {z:.6f}")
    m0 = np.array([r["m0"] for r in rows]); m1 = np.array([r["m0_knob"] for r in rows]); ts = np.array([r["top_same"] for r in rows])
    print(f"first look m0 {np.median(m0):+.2f}, with knob change {np.median(m1 - m0):+.2f}; top word same on last 200: median {np.median(ts):.3f}, mean {ts.mean():.3f}, lowest {ts.min():.3f}")
    a1, c1 = R["knob:1"]; a5 = R["knob:0.5"][0][0]; rnd = abs(R["random:1"][0][0])
    if rnd >= .5: v = "VOID, the random direction moved K"
    elif a1[0] >= 2 and a1[1] > 0 and c1[0] <= .25 and .35 <= a5 / a1[0] <= .65: v = "THE SONG TURNS UP"
    elif a1[0] >= .5 and a1[1] > 0: v = "ONLY A LITTLE"
    else: v = "NO KNOB HERE"
    print("half dose share of full dose:", round(a5 / a1[0], 2) if a1[0] else None)
    print("KNOB:", v)
    if ts.mean() < .95 or np.median(m1 - m0) < -1:
        print("DAMAGE: WRECKS HIM overrides, if mode is all")
    sys.exit()
if cmd == "c5":
    P = {r["step"]: r for r in load(sys.argv[2])}; Q = load(sys.argv[3]); Q = [r for r in Q if r["step"] in P]
    ids = np.array([r["id"] for r in Q]); print("steps", len(Q), "problems", len(set(ids)))
    K0 = np.array([KC(P[r["step"]]["reads"]["plain"])[0] for r in Q]); C0 = np.array([KC(P[r["step"]]["reads"]["plain"])[1] for r in Q])
    Kc = np.array([KC(r["reads"]["plain"])[0] for r in Q]); Cc = np.array([KC(r["reads"]["plain"])[1] for r in Q])
    a, b = boot(Kc - K0, ids), boot(Cc - C0, ids)
    print(f"c5 alone: dK {a[0]:+.2f} ({a[1]:+.2f} to {a[2]:+.2f})  dC {b[0]:+.2f} ({b[1]:+.2f} to {b[2]:+.2f})")
    print("COMPRESSOR:", "LEAVES THE SONG ALONE" if abs(a[0]) < .5 and abs(b[0]) < .5 else "IT MOVES THE SONG")
    Kp = np.array([KC(P[r["step"]]["reads"]["knob:1"])[0] for r in Q]) - K0
    Ks = np.array([KC(r["reads"]["knob:1"])[0] for r in Q]) - Kc
    Cs = np.array([KC(r["reads"]["knob:1"])[1] for r in Q]) - Cc
    x, y, c = boot(Kp, ids), boot(Ks, ids), boot(Cs, ids)
    print(f"knob rise on plain {x[0]:+.2f}, on c5 {y[0]:+.2f} ({y[1]:+.2f} to {y[2]:+.2f}); dC on c5 {c[0]:+.2f}")
    print("TOGETHER:", "ADDS IN CLEANLY" if x[0] and abs(y[0] - x[0]) <= .25 * abs(x[0]) else "THEY FIGHT")
