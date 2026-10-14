#!/bin/bash
# Speed of one or more brains on this machine, same settings for each: reading (pp512) and talking (tg128) at a
# fresh chat and 2,048 tokens in, 3 repetitions, cooled between files.
#   Mac (Metal):  ENGINE=.run/lcpp-src/build-mac/bin tools/squeeze/speed.sh a.gguf b.gguf
#   board (NPU):  run on the board with ENGINE=~/preview2/bin NPU=1
set -uo pipefail
ENGINE=${ENGINE:-.run/lcpp-src/build-mac/bin}
if [ "${NPU:-0}" = 1 ]; then
  export LD_LIBRARY_PATH=$(dirname $ENGINE)/lib ADSP_LIBRARY_PATH=$(dirname $ENGINE)/lib GGML_HEXAGON_USE_HMX=0 GGML_HEXAGON_OPBATCH=4096 LG_COOP=1
  DEV="-ngl 99 --device HTP0 -t 4 -C 0xf0 --cpu-strict 1"
else
  DEV="-ngl 99"
fi
for f in "$@"; do
  sleep ${COOL:-20}
  echo "== $(basename $f) ($(du -h "$f" | cut -f1))"
  $ENGINE/llama-bench --offline -m "$f" $DEV -fa 1 -p 512 -n 128 -d 0,2048 -r 3 -o md 2>&1 | grep -E "pp512|tg128" | awk -F'|' '{print "  " $(NF-2) $(NF-1)}'
done
