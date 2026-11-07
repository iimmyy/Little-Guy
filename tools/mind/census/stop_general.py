#!/usr/bin/env python3
"""Is LFM3's stop structure general? (Immy 2026-11-03: a principle, not one model's quirk; bare question only.)
On an untouched model's own runs (train split, bare), the chance of its end-of-thinking token at every line end after
the first right statement of the answer:
- finished runs: the last line (where it did stop) vs its earlier line ends. On greedy runs the last line is high by
  construction (we picked where it stopped), so the evidence is the earlier lines; on SAMPLED runs a stop can land on a
  low chance, so a high chance there is real evidence of one sharp switch.
- cut-off runs: every line end after the answer, and each run's highest chance.
- the kinds of last line: the two-word openers, and the chance at earlier lines with the same opener in the same runs
  (is it the kind of sentence, or the place it comes?).
usage: stop_general.py <lfm3|qwen|e4b> <hf folder or id> <out.txt> <runs.jsonl> [...] [n=40 dev=cuda]
"""
import json, os, re, sys
from collections import Counter
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natural_error import whole

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
FAM, path, out, runs = args[0], args[1], args[2], args[3:]
N, DEV = int(opt.get("n", 40)), opt.get("dev", "cuda")
if FAM == "lfm3":
    from patients import _lfm3
    p = _lfm3(DEV, path)
    tok, body, head_end, END_S, lm = p.tok, p.model.model, "\n", "</think>", p.model.lm_head
    prob = lambda h: torch.softmax(lm(h).float(), -1)
elif FAM == "qwen":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    mm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval().requires_grad_(False)
    body, head_end, END_S, lm = mm.model, "", "</think>", mm.lm_head
    prob = lambda h: torch.softmax(lm(h).float(), -1)
else:
    from patients import _e4b
    p = _e4b(DEV)
    tok, body, head_end, END_S = p.tok, p.model.model, "<|channel>thought\n", "<channel|>"
    prob = lambda h: torch.softmax(p.logits(h) if hasattr(p, "logits") else p.model.lm_head(h).float(), -1)
END = tok(END_S, add_special_tokens=False).input_ids
assert len(END) == 1, END
END = END[0]


def first_statement(think, A):
    mm_ = re.search(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b)\s*\$?\\?\(?\s*{A}(?![\d.,/])", think)
    return mm_.end() if mm_ else None


def head_of(t):
    return tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True) + head_end


def opener(line):
    return " ".join(re.sub(r"[*_#>\-\d.)(]+", " ", line).split()[:2])


def forward(ids):
    with torch.no_grad():
        o = body(input_ids=torch.tensor([ids], device=DEV))
    return o.last_hidden_state[0]


items = {"greedy": [], "sampled": []}
total = Counter()
for f in runs:
    kind = "sampled" if re.search(r"-s\d+\.jsonl$", f) else "greedy"
    for t in map(json.loads, open(f)):
        if FAM == "e4b" and not t["thinking"] and t["reply"].startswith("<|channel>thought"):
            t["thinking"] = t["reply"].replace("<|channel>thought", "", 1).strip()    # cut before the end tag: run_baseline left it all in the reply
        total[(kind, t["finish"], "thinking" if t["thinking"] else "no thinking")] += 1
        A = whole(str(t["answer"]))
        if A is None or A < 2 or not t["thinking"]:
            continue
        st = first_statement(t["thinking"], A)
        if st is not None:
            items[kind].append((t, st))
L = [f"stop structure, {FAM}, {os.path.basename(path.rstrip('/'))}, bare question. Runs: " + ", ".join(f"{k[0]} {k[1]} {k[2]} {v}" for k, v in sorted(total.items()))]
heads = Counter()
for kind, group in items.items():
    fin = [x for x in group if x[0]["finish"] == "stop"][:N]
    cut = [x for x in group if x[0]["finish"] != "stop" and END_S not in x[0]["reply"]][:N]
    if not group:
        continue
    joins = {}
    for j in ("", "\n", "\n\n"):
        pr = []
        for t, st in fin[:12]:
            ids = tok(head_of(t) + t["thinking"] + j, add_special_tokens=False).input_ids
            pr.append(float(prob(forward(ids)[-1])[END]))
        joins[j] = float(np.median(pr)) if pr else 0.0
    J = max(joins, key=joins.get)
    L.append(f"{kind}: {len(fin)} finished, {len(cut)} cut-off runs with the right answer stated; chance of the end token by join: "
             + ", ".join(f"{a!r} {b:.3f}" for a, b in joins.items()))
    stop, early, same, cutp, cutmax = [], [], [], [], []
    last_open = Counter()
    per = []
    for t, st in fin:
        th = t["thinking"]
        ends = [mm_.start() for mm_ in re.finditer(r"\n", th) if mm_.start() > st]
        head = head_of(t)
        enc = tok(head + th + J, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs, h0 = enc.input_ids, enc.offset_mapping, len(head)
        hs = forward(ids)
        last = th.rsplit("\n", 1)[-1]
        op = opener(last)
        last_open[op] += 1
        pts = []
        for e in ends:                                   # the token that ends this line's text, before the newline
            cand = [i for i, (a, b) in enumerate(offs) if b <= h0 + e and b > a]
            if cand:
                ln = th[:e].rsplit("\n", 1)[-1]
                pts.append((cand[-1], opener(ln)))
        pe = prob(hs[[k for k, _ in pts] + [len(ids) - 1]])[:, END].cpu().numpy()
        stop.append(float(pe[-1]))
        early += [float(x) for x in pe[:-1]]
        per.append((op, pts, pe[:-1]))
    tops = {o for o, _ in last_open.most_common(12) if o}
    for op, pts, pe in per:
        same += [float(x) for (k, o), x in zip(pts, pe) if o in tops]
    for t, st in cut:
        th = t["thinking"]
        ends = [mm_.start() for mm_ in re.finditer(r"\n", th) if mm_.start() > st]
        head = head_of(t)
        enc = tok(head + th, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs, h0 = enc.input_ids, enc.offset_mapping, len(head)
        hs = forward(ids)
        pts = []
        for e in ends:
            cand = [i for i, (a, b) in enumerate(offs) if b <= h0 + e and b > a]
            if cand:
                pts.append(cand[-1])
        if pts:
            pe = prob(hs[pts])[:, END].cpu().numpy()
            cutp += [float(x) for x in pe]
            cutmax.append(float(pe.max()))

    def row(name, v):
        v = np.array(v)
        if len(v):
            L.append(f"  {name}: n {len(v)}, median {np.median(v):.3f}, mean {v.mean():.3f}, over 0.1 {np.mean(v > 0.1):.1%}, over 0.5 {np.mean(v > 0.5):.1%}")
    row("finished, the line where it stopped", stop)
    row("finished, earlier line ends after the answer", early)
    row("finished, earlier lines with a last-line opener", same)
    row("cut off, line ends after the answer", cutp)
    row("cut off, each run's highest chance", cutmax)
    L.append("  last-line openers: " + ", ".join(f"{o!r} {c}" for o, c in last_open.most_common(16)))
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
