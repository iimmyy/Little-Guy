#!/bin/bash
# The fair baseline for the squeezed 31B: the standard community recipe. An importance matrix from public text
# (WikiText-2 test, never one of our eval texts), then llama.cpp's stock low-bit types quantized from the QAT bf16,
# plus one plain Q3_K_M without the matrix as a control. Each file scored with the same evals as ours.
set -uo pipefail
S=${S:-/scratch}; W=${W:-/mnt/squeeze}; . $S/venv/bin/activate
B=$S/src/build/bin; CFG=$S/squeeze/gemma4-31b.json; Q="python $S/squeeze/squeeze.py"
log() { echo "== $(date -u +%H:%M:%S) $*"; }
mkdir -p $W
[ -f $S/31b.imatrix ] || { log imatrix; $B/llama-imatrix -m $S/31b-q8_0.gguf -f $S/bench/wiki.test.raw \
    -o $S/31b.imatrix -c 512 --chunks 80 -t 16; }
for t in Q3_K_M IQ3_M Q3_K_S; do
  f=$W/31b-imat-$t.gguf
  [ -f $f ] || { log "quantize $t"; $B/llama-quantize --imatrix $S/31b.imatrix $S/31b-bf16.gguf $f $t 16 > /dev/null 2>&1; }
  log "score $t $(stat -c %s $f)"; $Q $CFG score $f
done
f=$W/31b-plain-Q3_K_M.gguf
[ -f $f ] || { log "quantize plain Q3_K_M"; $B/llama-quantize $S/31b-bf16.gguf $f Q3_K_M 16 > /dev/null 2>&1; }
log "score plain Q3_K_M $(stat -c %s $f)"; $Q $CFG score $f
log done
