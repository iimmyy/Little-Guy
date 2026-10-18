#!/usr/bin/env python3
"""One table of every generation score gen_bench.sh saved, read from the harness's own result files.

  GSM8K      8-shot CoT, flexible extract (last number in the answer); strict format in brackets
  HumanEval  instruct, pass@1
  IFEval     prompt-level strict / loose
  needle     mean of niah_single_1, niah_single_2, niah_multikey_1 at 4k and at 16k (100 haystacks each)
usage: gen_table.py <gen out dir> [brain order, comma separated]
"""
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
order = sys.argv[2].split(",") if len(sys.argv) > 2 else None


def latest(d):
    fs = sorted(d.glob("*/results_*.json"))
    return json.load(open(fs[-1]))["results"] if fs else None


brains = sorted({p.name.rsplit("-", 1)[0] for p in root.iterdir() if p.is_dir()})
if order:
    brains = [b for b in order if b in brains] + [b for b in brains if b not in order]
print("| brain | GSM8K (strict) | HumanEval | IFEval strict / loose | needle 4k | needle 16k |")
print("|---|---|---|---|---|---|")
for b in brains:
    s = latest(root / f"{b}-short") or {}
    cell = lambda t, k: (f"{100 * s[t][k]:.1f} ± {100 * s[t][k.replace(',', '_stderr,')]:.1f}"
                         if t in s and isinstance(s[t].get(k.replace(',', '_stderr,')), float) else
                         f"{100 * s[t][k]:.1f}" if t in s else "-")
    g = cell("gsm8k_cot", "exact_match,flexible-extract")
    gs = f"{100 * s['gsm8k_cot']['exact_match,strict-match']:.1f}" if "gsm8k_cot" in s else "-"
    h = cell("humaneval_instruct", "pass@1,create_test")
    i = (f"{100 * s['ifeval']['prompt_level_strict_acc,none']:.1f} / {100 * s['ifeval']['prompt_level_loose_acc,none']:.1f}"
         if "ifeval" in s else "-")
    needle = []
    for L in (4096, 16384):
        r = latest(root / f"{b}-niah{L}")
        v = [r[t][f"{L},none"] for t in ("niah_single_1", "niah_single_2", "niah_multikey_1")] if r else []
        needle.append(f"{100 * sum(v) / len(v):.1f}" if v else "-")
    print(f"| {b} | {g} ({gs}) | {h} | {i} | {needle[0]} | {needle[1]} |")
