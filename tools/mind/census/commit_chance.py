#!/usr/bin/env python3
"""How likely is he to START a commit sentence at each paragraph end? (2026-11-03, end_point.py: the thinking ends
with near certainty right after a commit sentence ("Thus final.", "I'll produce ... final line.", "So final line:
Answer: 33"), and almost never anywhere else; 93% of finished runs write one, 21% of cut-off runs.)
Commit openers = the two-token starts of his last sentences in finished train runs: strict (Thus final, So final,
Now produce, Now final, Thus end, Thus output, Thus answer, Proceed, Answer:, End with) and loose (+ I'll, We'll).
At paragraph ends sampled like done_fit.py (before the answer / wrap-up / middle / loop tail), the chance of each
opener = P(first) * P(second | first), read in one batched forward per point (the context plus each first token).
usage: commit_chance.py <model folder> <out.txt> <runs.jsonl> [...] [per=4 dev=cuda]
"""
import json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL
from natural_error import whole

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
mdir, out, runs = args[0], args[1], args[2:]
PER, DEV = int(opt.get("per", 4)), opt.get("dev", "cuda")
if opt.get("tail") == "0":                                # the bare question (Immy: no prompts)
    TAIL = ""
STRICT = [(21009, 2023), (4597, 2023), (8175, 4669), (8175, 2023), (21009, 1263), (21009, 6159), (21009, 5141),
          (94199, None), (39630, 34), (9219, 415)]                 # None: any second token
LOOSE = STRICT + [(49, 5515), (2083, 5515)]
p = _lfm3(DEV, mdir)
tok, m = p.tok, p.model
if opt.get("tail") == "0":                                # openers of his last sentences on bare questions (stop_general.py: untouched, 100 stops)
    one = lambda t_: tok(t_, add_special_tokens=False).input_ids[0]
    STRICT = [(one(a), one(b)) for a, b in (("Thus", " final"), ("So", " final"), ("Therefore", " final"), ("Hence", " final"), ("Thus", " answer"), ("So", " answer"),
                                            ("Therefore", " answer"), ("Thus", " \\"), ("Answer", ":"), ("Now", " produce"))]
    LOOSE = STRICT + [(one(a), one(b)) for a, b in (("I", "'ll"), ("We", "'ll"), ("We", " put"), ("Now", ","))]
FIRSTS = sorted({a for a, _ in LOOSE})


def first_statement(think, A):
    mm = re.search(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b)\s*\$?\\?\(?\s*{A}(?![\d.,/])", think)
    return mm.end() if mm else None


rng = np.random.RandomState(0)
res = {(k, g): {"strict": [], "loose": [], "first": [], "second": []} for k in ("handed in", "cut off") for g in range(4)}
NAMES = ["before the answer", "wrap-up (<200 after)", "middle (200-600)", "loop tail (>600)"]
nrun = 0
for f in runs:
    for t in map(json.loads, open(f)):
        A = whole(str(t["answer"]))
        if A is None or A < 2:
            continue
        th = t["thinking"]
        st = first_statement(th, A)
        if st is None:
            continue
        head = tok.apply_chat_template([{"role": "user", "content": t["q"] + TAIL}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) + "\n"
        enc = tok(head + th, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs, h0 = enc.input_ids, enc.offset_mapping, len(head)
        full = head + th
        ends = [i for i, (a, b) in enumerate(offs) if a >= h0 and "\n\n" in full[a:b]]
        pre = [i for i in ends if offs[i][0] < h0 + st]
        post = [i for i in ends if offs[i][0] >= h0 + st]
        pick = (rng.choice(pre, min(PER, len(pre)), replace=False).tolist() if pre else []) + \
               (rng.choice(post, min(2 * PER, len(post)), replace=False).tolist() if post else [])
        w_st = len(re.findall(r"\S+", th[:st]))
        nrun += 1
        kind = "handed in" if t["finish"] == "stop" else "cut off"
        for i in pick:
            aft = max(0, len(re.findall(r"\S+", th[:max(0, offs[i][1] - h0)])) - w_st) if offs[i][0] >= h0 + st else -1
            g = 0 if aft < 0 else 1 if aft < 200 else 2 if aft <= 600 else 3
            ctx = ids[:i + 1]
            batch = torch.tensor([ctx + [a] for a in FIRSTS], device=DEV)       # same length: one batched forward
            with torch.no_grad():
                hs = m.model(input_ids=batch).last_hidden_state
                p1 = torch.softmax(m.lm_head(hs[0, -2]).float(), -1)              # at the paragraph end itself
                p2 = torch.softmax(m.lm_head(hs[:, -1]).float(), -1)              # after each first token
            for nm, S in (("strict", STRICT), ("loose", LOOSE)):
                tot = 0.0
                for a, b in S:
                    k = FIRSTS.index(a)
                    tot += float(p1[a]) * (1.0 if b is None else float(p2[k, b]))
                res[(kind, g)][nm].append(tot)
            if opt.get("tail") == "0":                    # the two steps apart: the first word "Thus", and " final" after it
                th_ = FIRSTS.index(STRICT[0][0])
                res[(kind, g)]["first"].append(float(p1[STRICT[0][0]]))
                res[(kind, g)]["second"].append(float(p2[th_, STRICT[0][1]]))
        if nrun % 40 == 0:
            print(f"{nrun} runs", flush=True)
L = [f"commit chance at paragraph ends on {os.path.basename(mdir.rstrip('/'))}: {nrun} runs with the right answer stated"]
for (kind, g), r in res.items():
    if r["strict"]:
        s, l = np.array(r["strict"]), np.array(r["loose"])
        L.append(f"  {kind}, {NAMES[g]}: n {len(s)}, strict commit chance mean {s.mean():.3f} (median {np.median(s):.3f}, over 0.2 {np.mean(s > 0.2):.0%}), "
                 f"loose mean {l.mean():.3f}" + (f"; Thus {np.mean(r['first']):.3f}, final after Thus {np.mean(r['second']):.3f}" if r["first"] else ""))
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
