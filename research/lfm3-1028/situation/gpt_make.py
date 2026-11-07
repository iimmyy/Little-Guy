"""Ask the Azure gpt-4.1 deployment for new word problems with a sentence-by-sentence scene map (MathWorld style).
Key is read from az at run time, never written. usage: gpt_make.py <out.jsonl> <batch_n> <batches>"""
import json, subprocess, sys, urllib.request
out, per, nb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
az = lambda *a: subprocess.run(["az", *a, "-o", "tsv"], capture_output=True, text=True, check=True).stdout.strip()
AI = "lg-ai-" + next(l.split("=", 1)[1].strip() for l in open("littleguy.env") if l.startswith("LITTLEGUY_AZ_SUFFIX="))  # account suffix lives in littleguy.env
key = az("cognitiveservices", "account", "keys", "list", "-g", "littleguy", "-n", AI, "--query", "key1")
ep = az("cognitiveservices", "account", "show", "-g", "littleguy", "-n", AI, "--query", "properties.endpoint")
URL = ep.rstrip("/") + "/openai/deployments/judge/chat/completions?api-version=2024-10-21"
KINDS = ["rate (each X serves/costs/holds Y), like 'each meal feeds 1.5 people'", "transfer (gives, loses, buys, adds)",
         "compare (how many more / times as many)", "part-whole (groups that make a total)", "mixed: two of the kinds above"]
SYS = """You write brand-new grade-school math story problems for a test. They must be original: do not reuse or paraphrase
any problem from GSM8K, SVAMP, ASDiv, MAWPS, MATH or textbooks. Ordinary settings (kitchens, buses, gardens, shops, sports),
varied names, 2 to 4 sentences of story then one question. 2 to 4 arithmetic steps, whole-number answer.
For each problem give a scene map, sentence by sentence, in this JSON shape:
{"text": full problem text,
 "sentences": [{"span": exact sentence text,
                "things": [{"owner": who has it, "thing": what it is, "amount": number or "?", "unit": unit or ""}],
                "links": [{"kind": "rate|transfer|compare|partwhole", "from": thing, "to": thing, "value": number, "says": short plain meaning}]}],
 "asked": {"owner": ..., "thing": ..., "unit": ...},
 "equation": one python arithmetic expression using only the numbers in the text that equals the answer,
 "answer": the whole-number answer}
"things" lists every quantity that the scene holds after that sentence (amounts not yet known are "?").
Every number in the text must appear in the map. Return JSON: {"problems": [...]}"""
rows = []
for b in range(nb):
    kind = KINDS[b % len(KINDS)]
    body = {"messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": f"Write {per} problems. Main kind: {kind}. Batch {b + 1}: use different settings from earlier batches."}],
            "temperature": 0.9, "response_format": {"type": "json_object"}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"api-key": key, "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))
    ps = json.loads(r["choices"][0]["message"]["content"])["problems"]
    for i, p in enumerate(ps):
        rows.append({"id": f"gpt/{b}-{i}", "source": "gpt-4.1", "kind": kind, **p})
    print(f"batch {b + 1}/{nb}: {len(ps)} problems", flush=True)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), out)
