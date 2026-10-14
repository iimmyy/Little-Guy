#!/bin/bash
# One variant per group of 2-bit tensors, that group taken back to 4-bit from Google's file, scored against Q8_0.
set -uo pipefail
S=${S:-/scratch}; V=${V:-/mnt/variants}; B=$S/src/build/bin; . $S/venv/bin/activate; mkdir -p $V
OURS=$S/31b-C-preview.gguf; GOOGLE=$S/google/gemma-4-31B_q4_0-it.gguf
score() { LG_DICT_PREVIEW=0 $B/llama-perplexity -m "$1" -f $S/kl_text.txt -c 512 --chunks 4 -t 16 --kl-divergence-base $S/31b_kl_base.bin --kl-divergence 2>&1 | grep -E "Mean    KLD|Same top" | tr -s ' ' | tr '\n' ' '; }
for g in attn_q attn_k ffn_gate ffn_up; do
  f=$V/31b-solo-$g.gguf
  python $S/swap_tracks.py $S/src/gguf-py $OURS $GOOGLE $f "\.${g}\.weight$" > /dev/null
  echo "== $g at 4-bit ($(du -h $f | cut -f1)): $(score $f)"
  rm -f $f
done
