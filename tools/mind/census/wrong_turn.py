#!/usr/bin/env python3
"""Where does a run go wrong? (Immy 2026-10-30: walk back along the trail to the fork.) Good and bad runs of the same
question look the same inside for the first ~500 words (tightness.py), so the outcome is decided during the run.
For a bad run, cut its thinking at several sentence starts and let the untouched model finish from each cut K times
(same sampling as the split runs, same 6000-word budget counted from the start). The share of finishes that end
right is his chance of solving from that point ("value"; the words where it falls are critical tokens, Lin et al.
2411.19943). The fork lies between the last cut that still solves and the first that no longer does.

usage: wrong_turn.py <server url> <answers jsonl[,more]> <out.jsonl> [hf=~/.littleguy/hf/lfm3-1.7b nq=10 k=4
                     cuts=0,200,500,1000,1500,2000,3000,4000 par=4 only=<question id>]
Rows are written as they finish and skipped on a rerun (resume). report: wrong_turn.py report <out.jsonl>
"""
import json, os, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right, right_reply

TAIL = "\n\nEnd with a line: Answer: <your answer>"
BUDGET = 6000


def good_row(r):
    return r["finish"] == "stop" and bool(r["reply"]) and right_reply(r, r["reply"])


def report(path):
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        by.setdefault(r["id"], {}).setdefault(r["cut"], []).append(r)
    for q, cuts in by.items():
        cs = sorted(cuts)
        v = [sum(x["right"] for x in cuts[c]) / len(cuts[c]) for c in cs]
        print(f"\n{q}  (bad run seed {cuts[cs[0]][0]['src_seed']})")
        print("  cut word  " + "  ".join(f"{c:5d}" for c in cs))
        print("  value     " + "  ".join(f"{x:5.2f}" for x in v))
        hi = [i for i, x in enumerate(v) if x > 0]
        if hi and hi[-1] + 1 < len(cs):                       # last cut that still solves, then the first that doesn't
            a, b = cs[hi[-1]], cs[hi[-1] + 1]
            print(f"  fork between word {a} and {b}:")
            print("    " + cuts[b][0]["between"].replace("\n", " / ")[:900])
        elif not hi:
            print("  never solves from any cut (lost from the start, or too hard)")
        else:
            print("  still solves from the last cut")


if ARGS[0] == "report":
    report(ARGS[1])
    sys.exit()

from transformers import AutoTokenizer

url, srcs, out = ARGS[0], ARGS[1].split(","), ARGS[2]
opt = dict(a.split("=", 1) for a in ARGS[3:])
HF = os.path.expanduser(opt.get("hf", "~/.littleguy/hf/lfm3-1.7b"))
NQ, K, PAR = int(opt.get("nq", 10)), int(opt.get("k", 4)), int(opt.get("par", 4))
CUTS = [int(x) for x in opt.get("cuts", "0,200,500,1000,1500,2000,3000,4000").split(",")]
tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)

runs = {}
for s in srcs:
    seed = int(re.search(r"-s(\d+)\.jsonl$", s).group(1)) if re.search(r"-s(\d+)\.jsonl$", s) else 0
    for l in open(s):
        r = json.loads(l)
        if len((r["thinking"] or "").split()) > 50:
            runs.setdefault(r["id"], []).append((good_row(r), seed, r))
# questions he solves sometimes (1 to 3 of 4-5 runs): the fork is real there, not a question he can't do at all
mixed = sorted(q for q, rs in runs.items() if 0 < sum(g for g, _, _ in rs) < len(rs))
if "only" in opt:
    mixed = [opt["only"]]
random.Random(0).shuffle(mixed)
picked = mixed[:NQ]

jobs = []
for q in picked:
    bad = [(s, r) for g, s, r in runs[q] if not g]
    seed, r = bad[random.Random(q).randrange(len(bad))]
    prompt = tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    txt = "\n" + r["thinking"]
    enc = tok(txt, add_special_tokens=False, return_offsets_mapping=True)
    starts = [0] + [t for t in range(1, len(enc.input_ids))
                    if txt[max(0, enc.offset_mapping[t][0] - 2):enc.offset_mapping[t][0]].rstrip(" ").endswith(("\n", ".", "?", "!"))]
    prev = 0
    for c in CUTS:
        if c >= len(enc.input_ids):
            break
        t = next(x for x in starts if x >= c) if any(x >= c for x in starts) else None
        if t is None:
            break
        ch = enc.offset_mapping[t][0] if t else 0
        between = txt[prev:ch]
        prev = ch
        for k in range(K):
            jobs.append({"id": q, "cut": c, "tok": t, "k": k, "src_seed": seed, "row": r, "prefix": prompt + txt[:ch],
                         "used": len(enc.input_ids[:t]), "between": between})

done = set()
if os.path.exists(out):
    done = {(d["id"], d["cut"], d["k"]) for d in map(json.loads, open(out))}
todo = [j for j in jobs if (j["id"], j["cut"], j["k"]) not in done]
print(f"{len(picked)} questions, {len(jobs)} finishes ({len(todo)} to go)", flush=True)


def finish(j):
    body = json.dumps({"prompt": j["prefix"], "n_predict": max(200, BUDGET - j["used"]), "cache_prompt": True,
                       "temperature": 0.7, "top_p": 0.95, "top_k": 40,
                       "seed": 1000 * j["k"] + j["cut"] + 7}).encode()
    res = json.load(urllib.request.urlopen(urllib.request.Request(url + "/completion", body,
                                                                  {"Content-Type": "application/json"}), timeout=1800))
    text = res["content"]
    think, _, reply = text.partition("</think>") if "</think>" in text else (text, "", "")
    stopped = res.get("stop_type") in ("eos", "word") or res.get("stopped_eos")
    return {"id": j["id"], "cut": j["cut"], "tok": j["tok"], "k": j["k"], "src_seed": j["src_seed"],
            "finished": bool(stopped and reply.strip()), "between": j["between"], "rest": think[-1500:],
            "reply": reply.strip()}


with open(out, "a") as f, ThreadPoolExecutor(PAR) as ex:
    for n, (j, d) in enumerate(zip(todo, ex.map(finish, todo))):
        d["right"] = bool(d["finished"] and right_reply(j["row"], d["reply"]))     # main thread: math-verify uses signal.alarm
        f.write(json.dumps(d) + "\n"); f.flush()
        print(f"  {n + 1}/{len(todo)} {d['id']} cut {d['cut']} k {d['k']}: {'RIGHT' if d['right'] else 'wrong'}", flush=True)
print("TURN_DONE")
