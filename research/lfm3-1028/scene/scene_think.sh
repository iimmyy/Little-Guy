# Visualizer step 6 inputs: LFM3's own thinking on the 108 scene problems (pool minus sealed 300), plain and c5,
# stock llama-server, chat endpoint, greedy, 4 slots per server (same for both builds), cap 6000.
W=$HOME/work/ligands; N=$W/night; S=$W/scene; P=$S/scene-problems-108.jsonl
for name in plain c5; do
  $HOME/work/llama.cpp-lfm3/build/bin/llama-server -m $N/$name-BF16.gguf -ngl 99 -c 32768 --parallel 4 --port 8095 --host 127.0.0.1 --jinja --cache-ram 0 > $S/server-$name.log 2>&1 &
  pid=$!
  for i in $(seq 1 120); do curl -s -m 2 127.0.0.1:8095/health | grep -q ok && break; sleep 1; done
  echo "$(date +%H:%M) start $name (server $pid)"
  PYTHONPATH=$HOME/tf521 timeout 7200 $HOME/venv/bin/python $W/tools/mind/census/engine_run.py http://127.0.0.1:8095 $P $S/scene-think-$name.jsonl $HOME/.littleguy/hf/lfm3-1.7b chat=1 par=4 > $S/run-$name.log 2>&1
  kill $pid; sleep 3
  echo "$(date +%H:%M) done $name: $(wc -l < $S/scene-think-$name.jsonl) answers"
done
echo SCENE_THINK_DONE
