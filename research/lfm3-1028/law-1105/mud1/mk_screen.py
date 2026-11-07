import os
import json, hashlib, sys
W = os.path.expanduser("~") + "/work/ligands"
held = {json.loads(l)["id"] for l in open(f"{W}/science/room/mix/sci-held-81.jsonl")}
keys = json.load(open(f"{W}/science/grades-v4/keys-v2.json")) if False else None
h = lambda i: hashlib.md5(("screen1106" + i).encode()).hexdigest()
S = [json.loads(l) for l in open(f"{W}/science/science-build-all-v1.jsonl")]
S = [r for r in S if r["id"] not in held]
S.sort(key=lambda r: h(r["id"]))
M = [json.loads(l) for l in open(f"{W}/data/math500-night-train200.jsonl")]
M.sort(key=lambda r: h(r["id"]))
open("screen-sci-40.jsonl", "w").writelines(json.dumps(r) + "\n" for r in S[:40])
open("screen-math-40.jsonl", "w").writelines(json.dumps(r) + "\n" for r in M[:40])
print(len(S), len(M), S[0]["id"], M[0]["id"], list(M[0].keys())[:8])
