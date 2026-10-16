#!/bin/bash
# Capability benchmarks (log-likelihood scoring, llama-perplexity): HellaSwag (all 10,042), Winogrande (all 1,267),
# ARC-Challenge, MMLU, TruthfulQA (validation sets, all tasks). Full logs kept next to the summary.
# usage: tools/squeeze/bench_fresh.sh <out dir> <model.gguf> [...]
cd "$(dirname "$0")/../.."
B=.run/lcpp-src/build-mac/bin; O=$1; shift; mkdir -p $O; R=.run/bench
for m in "$@"; do
  n=$(basename $m .gguf)
  run() { LG_DICT_PREVIEW=0 $B/llama-perplexity -m $m -ngl 99 "$@" 2>&1; }
  run -f $R/hellaswag_val_full.txt --hellaswag --hellaswag-tasks 10042 > $O/$n.hellaswag.log
  run -f $R/winogrande-debiased-eval.csv --winogrande > $O/$n.winogrande.log
  for t in arc-challenge-validation mmlu-validation truthful-qa-validation; do
    run -bf $R/$t.bin --multiple-choice > $O/$n.$t.log
  done
  echo "$n"
  for f in $O/$n.*.log; do echo "  $(basename $f .log | sed "s/^$n\.//"): $(grep -E "^[0-9]+\s+[0-9.]+|Final|score" $f | tail -1)"; done
done
