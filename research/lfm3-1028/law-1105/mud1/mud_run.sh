#!/bin/sh
# Untouched then c5, reading only. usage: mud_run.sh
W=$HOME/work/ligands; O=$W/science/room/mud; PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python -u"
cd $O
$PY mud_read.py $HOME/.littleguy/hf/lfm3-1.7b votes-steps-v1.jsonl plain > mud-plain.log 2>&1
$PY mud_read.py $W/pam/c5 votes-steps-v1.jsonl c5 > mud-c5.log 2>&1
echo MUD_RUN_DONE
