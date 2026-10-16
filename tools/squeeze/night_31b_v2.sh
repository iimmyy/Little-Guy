#!/bin/bash
# Squeeze v2 on the 31B, one resumable run: donors -> Q8 reference on the public held-out text -> harm map ->
# calibration probes -> quality-vs-size curve, plus the community files on the same ruler.
set -uo pipefail
S=${S:-/scratch}; . $S/venv/bin/activate
B=$S/src/build/bin; CFG=$S/squeeze/gemma4-31b-v2.json; Q="python $S/squeeze/squeeze2.py $CFG"; M=/mnt/squeeze
log() { echo "== $(date -u +%H:%M:%S) $*"; }
until grep -q "== .* done" $S/donors_v2.log; do sleep 60; done; log "donors ready"
[ -f $S/31b_q8_wiki512.kl ] || { log "Q8 reference"; $B/llama-perplexity -m $S/31b-q8_0.gguf -f $S/bench/wiki.eval.txt \
    -c 512 --chunks 8 -t 16 --kl-divergence-base $S/31b_q8_wiki512.kl > $S/ref512.log 2>&1; }
log discover; $Q discover
log harm; $Q harm
log calibrate; $Q calibrate
for t in Q3_K_S IQ3_M Q3_K_M; do   # community files on the same ruler (rebuilt: /mnt was wiped by the restart)
  f=$M/31b-imat-$t.gguf
  [ -f $M/v2/community-$t.json ] && continue
  log "community $t"; $B/llama-quantize --imatrix $S/31b.imatrix $S/31b-bf16.gguf $f $t 16 > /dev/null 2>&1
  echo "{\"bytes\": $(stat -c %s $f), \"score\": $($Q score $f)}" > $M/v2/community-$t.json; rm -f $f
  cat $M/v2/community-$t.json
done
log curve; $Q curve 13.8 14.4 15.3 16.5 17.2
log done
