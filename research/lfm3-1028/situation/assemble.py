"""Shuffle 50 hand-checked MathWorld + 50 hand-checked gpt-4.1 problems into one set (seed 1102). Hand-read lists below."""
import json, random, re
MW_DROP = {"mw/mawps-194", "mw/svamp-153", "mw/svamp-302", "mw/svamp-81"}      # wrong key, fragment, impossible, absurd
GPT2_DROP = {"0-2", "0-6", "0-10", "1-3", "3-1", "3-5", "4-11",                   # broken logic or wrong answer
             "3-0", "3-2", "3-3", "3-4", "3-6", "3-7", "3-9", "3-10", "3-11"}     # sum wrong in gpt's answer (not hand-fixed)
GPT1_TAKE = {"4-0", "4-1", "4-2", "4-5", "4-6", "4-7"}                           # first batch: the multi-step ones, all right
mw, seen = [], set()
for r in map(json.loads, open("mw-pool-v1.jsonl")):
    tmpl = " ".join(re.sub(r"[^a-z ]", "", r["text"].lower()).split()[:4])
    if r["id"] in MW_DROP or tmpl in seen:
        continue
    seen.add(tmpl); mw.append(r)
gpt = [dict(r, id="gpt/v2-" + r["id"][4:]) for r in map(json.loads, open("gpt-problems-raw-v2.jsonl")) if r["id"][4:] not in GPT2_DROP]
gpt += [dict(r, id="gpt/v1-" + r["id"][4:]) for r in map(json.loads, open("gpt-problems-raw-v1.jsonl")) if r["id"][4:] in GPT1_TAKE]
print("mathworld usable", len(mw), "gpt usable", len(gpt))
rng = random.Random(1102)
rng.shuffle(mw)
S = mw[:50] + gpt[:50]
rng.shuffle(S)
with open("situation-100-v1.jsonl", "w") as f:
    for i, r in enumerate(S):
        f.write(json.dumps({"n": i, "id": r["id"], "source": r["source"], "text": r["text"], "answer": int(r["answer"]),
                            "sentences": r["sentences"]}) + "\n")
print("wrote", len(S), "| first 8 sources:", [r["source"] for r in S[:8]])
