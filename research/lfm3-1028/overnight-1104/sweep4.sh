#!/bin/sh
# Immy's target is more gain; the spill (clock closes with no answer) is the leak. Earlier closes: does the reply do the work for less? Only the clock changes, stop cell and formula stay. n=name floating forced.
B=/opt/dlami/nvme/big; D=$HOME/work/ligands/brake/scene; N=$HOME/work/ligands/night; C=$HOME/work/ligands/tools/mind/census; E=../eval-107.jsonl
cd $D
PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python"
RUNS="$N/train-c5-bare.jsonl $N/train-c5-bare-s1.jsonl $N/train-plain-bare.jsonl $N/train-plain-bare-s1.jsonl"
[ -f cap-f2d.npz ] || $PY $C/done_cap.py $B/f2-s cap-f2d.npz $RUNS skip=$E > cap-f2d.log 2>&1
p=8120
for v in "f3 1000 700" "f4 700 500"; do
  set -- $v
  $PY $C/clock_cell.py $B/f2-d cap-f2d.npz $B/$1 $N/train-c5-bare.jsonl $N/train-c5-bare-s1.jsonl skip=$E N=$2 Nf=$3 mat=mat.npz > $1-u.log 2>&1
  cd $B; env PYTHONPATH=$HOME/tf521:$HOME/work/llama.cpp-lfm3/gguf-py $HOME/venv/bin/python $HOME/work/llama.cpp-lfm3/convert_hf_to_gguf.py $1 --outtype bf16 --outfile $1-BF16.gguf > $1-gguf.log 2>&1
  cd $HOME/work/ligands/brake; sh eval_box1.sh $1 $B/$1-BF16.gguf $p > eval-$1.log 2>&1
  p=$((p+1)); cd $D
done
touch $B/sweep4.done
