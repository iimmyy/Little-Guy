"""Keep the hand-read 100 and 20 exactly; swap in the fuller MathWorld maps (kinds + numbers on links, mw_pick.py v2)."""
import json, sys
new = {r["id"]: r for r in map(json.loads, open("mw-all-v2.jsonl"))}
for src, out in (sys.argv[1], "situation-100-v1.jsonl"), (sys.argv[2], "situation-broken-v1.jsonl"):
    rows = list(map(json.loads, open(src)))
    for r in rows:
        if r["source"] == "mathworld":
            assert new[r["id"]]["text"] == r["text"], r["id"]
            r["sentences"] = new[r["id"]]["sentences"]
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(out, len(rows), "mathworld remapped", sum(r["source"] == "mathworld" for r in rows))
