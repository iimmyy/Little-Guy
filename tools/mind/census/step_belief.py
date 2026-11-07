#!/usr/bin/env python3
"""Does the compressor's key already read belief at middle STEPS, not just at the final answer? (Immy 2026-11-02, option B)

The c5 key is one layer-17 neuron: gate = 30 (belief - 0.85), up = belief, both straight-line reads of layer 17's MLP
input. Layers 0-16 and layer 17's mixer are untouched by the build, so the untouched model gives the key's exact input on
any text; the key's two rows come from the built file (key_rows.npz, pulled from the c5 GGUF).

Checkable steps: plain arithmetic in the thinking ("a op b = c", numbers only), graded by computing it. Each step is
read at its fork, the word before the next sentence starts (where the key acts). Also tagged: does the next sentence
doubt (But / However / Wait / Actually / Let's check ...)?

usage: step_belief.py <answers.jsonl[,more]> <key_rows.npz> <out.txt> [hf=<folder>] [dev=mps]
"""
import json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

files, rows_npz, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
DEV, KL, CTH = opt.get("dev", "mps"), 17, 0.85
TAIL = "\n\nEnd with a line: Answer: <your answer>"
# 2026-11-02 fix (the first parse read its own mistakes as his: read by hand): whole numbers only, thousands commas
# kept together, the step must stand alone (nothing operator-like or word-like right before a or right after c), and
# no rounding ("≈", "about", decimals). Division counts only when it comes out whole.
NUM = r"\d{1,3}(?:,\d{3})+(?!\d)|\d+"
ARITH = re.compile(rf"({NUM})\s*([+\-*×/]|\\times|\\cdot)\s*({NUM})\s*=\s*({NUM})")
DOUBT = re.compile(r"\W*(but|however|wait|hmm|actually|hold on|let'?s (double[- ]?check|check|verify|recompute|re-?check)|"
                   r"let me (check|verify|recompute)|double[- ]check|check:|verify)\b", re.I)


def steps_in(text):
    """yield (match, right) for every stand-alone whole-number step "a op b = c" in the text"""
    for mm in ARITH.finditer(text):
        before = text[:mm.start()].rstrip(" ")[-1:]
        after = text[mm.end():mm.end() + 3]
        if before and (before.isalnum() or before in "+-*×/^(.,=·√≈\\_{"):
            continue                                            # a piece of a longer expression (or a decimal)
        if re.match(r"\s*([\w(^/*×+\-.]|\\[a-z]|,\d)", after) and not re.match(r"\.(\s|$)", after):
            continue                                            # the result runs on: "= 2.5", "= 4 + ..", "= 12x"
        if re.search(r"(≈|~|about|approx|roughly)\s*$", text[max(0, mm.start() - 12):mm.start()], re.I):
            continue
        a, op, b, c = (int(mm[1].replace(",", "")), {"\\times": "*", "\\cdot": "*", "×": "*"}.get(mm[2], mm[2]),
                       int(mm[3].replace(",", "")), int(mm[4].replace(",", "")))
        if op == "/" and (b == 0 or a % b):
            continue
        yield mm, {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else None}[op] == c


if opt.get("parse_only") == "1":                               # read the wrong ones by hand before any model run
    n = w = 0
    for f in files:
        for r in map(json.loads, open(f)):
            for mm, ok in steps_in("\n" + r["thinking"]):
                n += 1
                if not ok:
                    w += 1
                    t = "\n" + r["thinking"]
                    print(f"WRONG {r['id'].split('test/')[-1]}: {t[max(0, mm.start() - 60):mm.end() + 25]!r}")
    print(f"steps {n}, wrong {w}")
    sys.exit()

p = _lfm3(DEV, opt.get("hf"))
tok, m = p.tok, p.model
K = np.load(rows_npz)
up = torch.tensor(K["c5_up"], device=DEV, dtype=torch.float32)
gate = torch.tensor(K["c5_gate"], device=DEV, dtype=torch.float32)
cap = {}
m.model.layers[KL].feed_forward.register_forward_pre_hook(lambda mod, a: cap.__setitem__("x", a[0][0].float()))

if opt.get("slip") == "1":
    # Natural wrong steps are too rare to test (1 in 297), so make them: every right step is read twice on the text up to
    # the end of its sentence, once as written and once with only its result's last digit slipped by one ("12*7=85").
    # Same words otherwise, so a belief drop is the step being wrong, nothing else.
    P = []
    for f in files:
        for r in map(json.loads, open(f)):
            pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
            think = "\n" + r["thinking"]
            cut = len(tok.decode(tok(think, add_special_tokens=False).input_ids[:6000]))
            for mm, ok in steps_in(think[:cut]):
                if not ok:
                    continue
                e = mm.end()
                while e < cut and not (think[e] == "\n" or think[e] == "." and (e + 1 >= len(think) or think[e + 1] in " \n")):
                    e += 1
                c = mm[4]
                d = int(c[-1])
                bad = c[:-1] + str(d + 1 if d < 9 else d - 1)
                pair = []
                for txt in (think[:e + 1], think[:mm.start(4)] + bad + think[mm.end(4):e + 1]):
                    with torch.no_grad():
                        m.model(input_ids=torch.tensor([pr + tok(txt, add_special_tokens=False).input_ids], device=DEV))
                    pair.append(float(cap["x"][-1] @ up))
                P.append((os.path.basename(f), r["id"], mm[0], bad, *pair))
            print(f"{os.path.basename(f)} {r['id']} pairs so far {len(P)}", flush=True)
    A = np.array([p[4:] for p in P])
    d = A[:, 0] - A[:, 1]
    L = [f"slip test (c5 key belief, layer 17 #1076, threshold {CTH}), {len(P)} right steps, each also read with its result slipped by one digit",
         f"  as written: belief median {np.median(A[:, 0]):+.2f}, over threshold {100 * np.mean(A[:, 0] > CTH):.1f}%",
         f"  slipped:    belief median {np.median(A[:, 1]):+.2f}, over threshold {100 * np.mean(A[:, 1] > CTH):.1f}%",
         f"  drop (written - slipped): median {np.median(d):+.3f} (10-90% {np.percentile(d, 10):+.3f}..{np.percentile(d, 90):+.3f}); "
         f"written higher in {100 * np.mean(d > 0):.1f}% of pairs",
         f"  slipped steps the key would still wave through (over threshold): {int(np.sum(A[:, 1] > CTH))}"]
    open(out, "w").write("\n".join(L) + "\n")
    json.dump(P, open(out.replace(".txt", ".json"), "w"))
    print("\n".join(L))
    sys.exit()


steps = []          # (run file, id, finished, right, doubted next, belief, key fires)
fork_all = []       # belief at every sentence-end fork (baseline)
for f in files:
    for r in map(json.loads, open(f)):
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        think = "\n" + r["thinking"]
        enc = tok(think, add_special_tokens=False, return_offsets_mapping=True)
        th, starts = enc.input_ids[:6000], np.array([s for s, _ in enc.offset_mapping[:6000]])
        with torch.no_grad():
            m.model(input_ids=torch.tensor([pr + th], device=DEV))
        x = cap["x"][len(pr):]
        bel = (x @ up).cpu().numpy()
        fire = (F.silu(x @ gate) * (x @ up)).cpu().numpy()
        lim = int(enc.offset_mapping[min(len(th), 6000) - 1][1])

        def fork(e):        # word before the next sentence: the sentence end at or after e, then the next start
            while e < lim and not (think[e] == "\n" or think[e] == "." and (e + 1 >= len(think) or think[e + 1] in " \n")):
                e += 1
            s = e + 1
            while s < lim and think[s] in " \n":
                s += 1
            t = int(np.searchsorted(starts, s, side="right")) - 1
            return (t - 1, s) if 1 <= t < len(th) and s < lim - 2 else (None, s)

        seen = set()
        for mm, ok in steps_in(think[:lim]):
            t, s = fork(mm.end())
            if t is None or t in seen:
                continue
            seen.add(t)
            steps.append((os.path.basename(f), r["id"], r["finish"] == "stop", ok, bool(DOUBT.match(think[s:s + 40])),
                          float(bel[t]), float(fire[t])))
        for mm in re.finditer(r"\.\s+(?=[A-Z])|\n", think[:lim]):
            t, _ = fork(mm.start())
            if t is not None and t not in seen:
                fork_all.append(float(bel[t]))
        print(f"{os.path.basename(f)} {r['id']} tokens {len(th)} steps so far {len(steps)}", flush=True)

S = np.array([(s[2], s[3], s[4], s[5], s[6]) for s in steps], dtype=float)
L = [f"step belief (c5 key, layer 17 #1076, threshold {CTH}); files {', '.join(os.path.basename(f) for f in files)}",
     f"baseline: every sentence-end fork in the thinking, n {len(fork_all)}: belief median {np.median(fork_all):+.2f}, "
     f"over {CTH}: {100 * np.mean(np.array(fork_all) > CTH):.1f}%"]
q = lambda v: f"median {np.median(v):+.2f} (10-90% {np.percentile(v, 10):+.2f}..{np.percentile(v, 90):+.2f})"
for name, sel in [("right steps", S[:, 1] == 1), ("wrong steps", S[:, 1] == 0),
                  ("right steps, next sentence doubts", (S[:, 1] == 1) & (S[:, 2] == 1)),
                  ("right steps, next sentence goes on", (S[:, 1] == 1) & (S[:, 2] == 0)),
                  ("right steps in cut-off runs", (S[:, 1] == 1) & (S[:, 0] == 0)),
                  ("right steps in finished runs", (S[:, 1] == 1) & (S[:, 0] == 1))]:
    v = S[sel]
    if len(v):
        L.append(f"{name:38s} n {len(v):5d}  belief {q(v[:, 3])}  key over threshold {100 * np.mean(v[:, 3] > CTH):5.1f}%  "
                 f"next sentence doubts {100 * np.mean(v[:, 2]):4.1f}%")
w = [s for s in steps if not s[3]]
L.append("wrong steps (for reading): " + "; ".join(f"{s[1].split('test/')[-1]} bel {s[5]:+.2f}" for s in w[:40]))
open(out, "w").write("\n".join(L) + "\n")
json.dump(steps, open(out.replace(".txt", ".json"), "w"))
print("\n".join(L))
