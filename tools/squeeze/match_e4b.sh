#!/bin/bash
# Output matching on E4B (Mac): record inputs, make stock donors with the importance matrix, match them, keep the
# per-matrix leftover error (.json next to each matched file) for allocation. Resumable: finished files are skipped.
# usage: tools/squeeze/match_e4b.sh <work dir> <bf16 master> [types...]
set -uo pipefail
cd "$(dirname "$0")/../.."
D=$1; REF=$2; shift 2; TYPES=${*:-Q2_K Q3_K Q4_0}
B=.run/lcpp-src/build-mac/bin; TXT=.run/bench/wiki.test.raw
log() { echo "== $(date +%H:%M:%S) $*"; }
mkdir -p $D/act
[ -f $D/e4b-q8_0.gguf ] || { log q8; $B/llama-quantize $REF $D/e4b-q8_0.gguf Q8_0 > /dev/null 2>&1; }
[ -f $D/e4b.imatrix ] || { log "imatrix + inputs"; LG_ACT_DUMP=$D/act LG_ACT_MAX=16384 $B/llama-imatrix -m $D/e4b-q8_0.gguf \
    -f $TXT -o $D/e4b.imatrix -c 512 --chunks 80 -ngl 99 > $D/imatrix.log 2>&1; }
for t in $TYPES; do
  [ -f $D/e4b-pure-$t.gguf ] || { log "donor $t"; $B/llama-quantize --pure --imatrix $D/e4b.imatrix $REF $D/e4b-pure-$t.gguf $t > /dev/null 2>&1; }
done
for t in $TYPES; do   # two at a time: each uses ~4 cores and ~7 GB
  [ -f $D/e4b-match-$t.gguf.json ] || { log "match $t"; python3 tools/squeeze/match.py build $REF $D/e4b-pure-$t.gguf $D/act $D/e4b-match-$t.gguf > $D/match-$t.log 2>&1 & }
  while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 10; done
done
wait; log done
