#!/bin/sh
# Waits for E4B bare to finish, then runs the average-write swap check on E4B (9 blocks) and LFM3 (each neuron block).
cd $HOME/work/ligands/science/room/fix
PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python"
for i in $(seq 1 360); do grep -q SE1_DONE chain7.log && break; sleep 10; done
echo "$(date -u +%H:%M) start"
timeout 1500 $PY g_lesion.py e4b $HOME/.littleguy/hf/e4b-full flip-steps-v1.jsonl combo-mean-e4b-9-v1.json combo=ffn:1,2,3,4,35,36,37,38,39 fill=mean
timeout 1500 $PY g_lesion.py lfm3 $HOME/.littleguy/hf/lfm3-1.7b votes-steps-v1.jsonl lesion-mean-lfm3-v1.json modes=ffn fill=mean
echo "$(date -u +%H:%M) CHAIN8_END"
