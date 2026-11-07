#!/bin/sh
O=$HOME/work/ligands/science/gain;PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python"
$PY $O/gain_read.py $HOME/.littleguy/hf/lfm3-1.7b $O/jobs3-v1.jsonl $O/plain > $O/plain.log 2>&1
[ -d /opt/dlami/nvme/big/f5 ] && $PY $O/gain_read.py /opt/dlami/nvme/big/f5 $O/jobs3-v1.jsonl $O/f5 > $O/f5.log 2>&1
touch $O/gain.done
