#!/bin/sh
# wire.sh <name> <base compressor> <lines17> <N floating> <N forced>: formula + stop + clock on one compressor, then gguf and eval on the problems not used for any of it
n=$1; base=$2; ln=$3; NW=$4; NF=$5
B=/opt/dlami/nvme/big; D=$HOME/work/ligands/brake/scene; N=$HOME/work/ligands/night; C=$HOME/work/ligands/tools/mind/census; E=../eval-107.jsonl
cd $D
PY="env PYTHONPATH=$HOME/tf521 $HOME/venv/bin/python"
RUNS="$N/train-c5-bare.jsonl $N/train-c5-bare-s1.jsonl $N/train-plain-bare.jsonl $N/train-plain-bare-s1.jsonl"
$PY $C/scene_cell.py $base $ln mat.npz $B/$n-s runs.jsonl > $n-s.log 2>&1 || exit 1
$PY $C/done_cap.py $B/$n-s cap-$n.npz $RUNS skip=$E > cap-$n.log 2>&1 || exit 1
$PY $C/done_cell.py $B/$n-s cap-$n.npz $B/$n-d $N/train-c5-bare.jsonl $N/train-c5-bare-s1.jsonl skip=$E gain=4 mat=mat.npz > $n-d.log 2>&1 || exit 1
$PY $C/clock_cell.py $B/$n-d cap-$n.npz $B/$n $N/train-c5-bare.jsonl $N/train-c5-bare-s1.jsonl skip=$E N=$NW Nf=$NF mat=mat.npz > $n-u.log 2>&1 || exit 1
rm -f cap-$n.npz
cd $B; env PYTHONPATH=$HOME/tf521:$HOME/work/llama.cpp-lfm3/gguf-py $HOME/venv/bin/python $HOME/work/llama.cpp-lfm3/convert_hf_to_gguf.py $n --outtype bf16 --outfile $n-BF16.gguf > $n-gguf.log 2>&1
cd $HOME/work/ligands/brake; sh eval_box1.sh $n $B/$n-BF16.gguf ${6:-8110} > eval-$n.log 2>&1
touch $B/$n.wired
