#!/bin/sh
W=$HOME/work/ligands;O=$W/science/models1;P=8300
$HOME/work/llama.cpp-lfm3/build-a10/bin/llama-server -m $W/night/plain-BF16.gguf -ngl 99 -c 196608 --parallel 16 --port $P --host 127.0.0.1 --jinja --cache-ram 0 > $O/server-lfm.log 2>&1 &
SV=$!
i=0;until curl -s -m 3 http://127.0.0.1:$P/health | grep -q ok;do i=$((i+1));[ $i -gt 120 ]&&{ kill $SV;exit 1;};sleep 5;done
$HOME/venv/bin/python $O/eq_test.py http://127.0.0.1:$P $O/lfm-eq-v1.jsonl runs=8 par=16 maxnew=8000 > $O/lfm-eq-v1.log 2>&1
kill $SV;touch $O/lfm.done
