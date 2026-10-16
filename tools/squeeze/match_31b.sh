#!/bin/bash
# 31B output matching on the Mac: Q3_K donor from the mixed-calibration importance matrix, matched on the mixed
# recorded inputs (damping 0.1) in two shards that share a cache, then one pass assembles the file.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=/Volumes/Music/LittleGuy/weights/big31b; B=.run/lcpp-src/build-mac/bin; REF=$D/31b-bf16.gguf
log() { echo "== $(date +%H:%M:%S) $*"; }
[ -f $D/31b-mixim-Q3_K.gguf ] || { log donor; $B/llama-quantize --pure --imatrix $D/31b-mix.imatrix $REF $D/31b-mixim-Q3_K.gguf Q3_K 12 > $D/q3k.log 2>&1; }
export LG_MATCH_DAMP=0.1 LG_MATCH_CACHE=$D/match-cache
for i in 0 1; do LG_MATCH_SHARD=$i/2 python3 tools/squeeze/match.py build $REF $D/31b-mixim-Q3_K.gguf $D/act_mix $D/31b-mm-Q3_K.gguf > $D/match-shard$i.log 2>&1 & done
wait; log assemble
python3 tools/squeeze/match.py build $REF $D/31b-mixim-Q3_K.gguf $D/act_mix $D/31b-mm-Q3_K.gguf > $D/match-assemble.log 2>&1
tail -1 $D/match-assemble.log; log done
