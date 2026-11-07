"""Human-written problems only (Immy 2026-11-02: gpt's problems follow a predictable AI format, scrap them). Every
MathWorld problem read by hand in mw-pool-read.txt, minus the 4 bad ones (kept in the broken set), one per template.
Maps from mw-all-v2.jsonl (kinds + numbers on links)."""
import json, re
DROP = {"mw/mawps-194", "mw/svamp-153", "mw/svamp-302", "mw/svamp-81"}
read = [l.split("]")[0][1:] for l in open("mw-pool-read.txt") if l.startswith("[")]
new = {r["id"]: r for r in map(json.loads, open("mw-all-v2.jsonl"))}
rows, seen = [], set()
for i in read:
    r = new[i]
    tmpl = " ".join(re.sub(r"[^a-z ]", "", r["text"].lower()).split()[:4])
    if i in DROP or tmpl in seen:
        continue
    seen.add(tmpl)
    rows.append({"n": len(rows), "id": i, "source": "mathworld", "text": r["text"], "answer": r["answer"], "sentences": r["sentences"]})
with open("situation-human-v1.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("hand-read", len(read), "kept", len(rows))
