"""MathWorld test-split problems into our scene-map shape; keep the multi-step ones. usage: mw_pick.py <data dir> <out.jsonl> <n>"""
import json, glob, os, random, sys
root, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for f in sorted(glob.glob(f"{root}/*/*/*.json")):
    G = json.load(open(f))
    if any(g["graph"]["metadata"].get(k) for g in G for k in ("flagged", "low confidence")):
        continue
    sents, ans, steps = [], None, 0
    for g in G:
        g = g["graph"]; nodes = g["nodes"]
        things = []
        for k, v in nodes.items():
            m = v["metadata"]
            if k == "CQ":
                ans = m["quantity"]
                continue
            q = m["quantity"]
            things.append({"owner": v["label"], "thing": " ".join((m["attribute"] + " " + m["entity"]).split()),
                           "amount": "?" if q.strip().startswith("x") else q.strip(), "unit": m["unit"]})
        links = []
        for e in g["edges"]:                    # numbers on links: rate X1 of X2 per X3; transfer/compare X1 of X2 for owner X3
            md = e["metadata"]
            a = md.get("X2", [""])[0] if isinstance(md.get("X2"), list) else ""
            b = md["X3"][0] if isinstance(md.get("X3"), list) else md.get("X3", "")
            links.append({"kind": e["relation"], "from": a or nodes[e["source"]]["metadata"]["entity"],
                          "to": b or nodes[e["target"]]["metadata"]["entity"], "value": str(md.get("X1", "")).strip()})
        sents.append({"span": g["metadata"]["text span"], "things": things, "links": links})
    steps = len({(l["kind"], l["to"]) for s in sents[-1:] for l in s["links"]})
    try:
        a = float(ans)
    except (TypeError, ValueError):
        continue
    if a != int(a) or (len(sents) < 3 and not os.environ.get("ALL")):
        continue
    rows.append({"id": "mw/" + G[0]["graph"]["id"], "source": "mathworld", "text": " ".join(s["span"] for s in sents),
                 "sentences": sents, "answer": int(a), "steps": steps})
if os.environ.get("ALL"):                    # every usable problem (for re-mapping a set already picked)
    with open(out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    sys.exit(print("all", len(rows)))
rows.sort(key=lambda r: -r["steps"])
pool = [r for r in rows if r["steps"] >= 2]
random.Random(1102).shuffle(pool)
pick = pool[:n]
print(f"usable {len(rows)}, with 2+ links at the end {len(pool)}, picked {len(pick)}")
with open(out, "w") as fo:
    for r in pick:
        fo.write(json.dumps(r) + "\n")
