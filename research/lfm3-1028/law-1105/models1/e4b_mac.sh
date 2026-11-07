#!/bin/sh
O=$LITTLEGUY_ROOT/.run/results/mind/lfm3-1028/law-1105/models1;P=8302
~/.littleguy/src/llama.cpp-lfm3/build/bin/llama-server -m $LITTLEGUY_WEIGHTS/fresh/e4b-full-bf16.gguf -ngl 99 -c 32768 --parallel 4 --port $P --host 127.0.0.1 --jinja --cache-ram 0 --reasoning-format none > $O/server-e4b.log 2>&1 &
SV=$!;echo $SV > $O/server-e4b.pid
i=0;until curl -s -m 3 http://127.0.0.1:$P/health | grep -q ok;do i=$((i+1));[ $i -gt 120 ]&&{ kill $SV;exit 1;};sleep 5;done
python3 $O/eq_test.py http://127.0.0.1:$P $O/e4b-eq-v1.jsonl runs=4 par=4 maxnew=6000 > $O/e4b-eq-v1.log 2>&1
kill $SV;touch $O/e4b.done
