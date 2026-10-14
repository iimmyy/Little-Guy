#!/bin/bash
# Public benchmark files for llama-perplexity, pinned by dataset commit, with checksums (bench.sha256).
# usage: get_bench.sh <dir>
set -euo pipefail
D=${1:-bench}; mkdir -p "$D"; cd "$D"
V=https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/37884b81b4957f1950a53b6ff48d77c8dd5e430c
W=https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/resolve/93f7fb0697ca6cd2c7e5727dcba3b7200fa4014c
H=https://raw.githubusercontent.com/klosax/hellaswag_text_data/1105d370f8340eac3c63a91fd1fa58e8999f9505
get() { [ -f "$2" ] || curl -sfL "$1" -o "$2"; }
get $H/hellaswag_val_full.txt hellaswag_val_full.txt
get $W/winogrande-debiased-eval.csv winogrande-debiased-eval.csv
for f in arc-challenge-validation.bin mmlu-validation.bin truthful-qa-validation.bin wiki.test.raw.gz; do get $V/$f $f; done
[ -f wiki.test.raw ] || gunzip -k wiki.test.raw.gz
