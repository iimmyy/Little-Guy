#!/bin/bash
# Can the board's 4 big cores read the below-2-bit formats as fast as Q2_K? They pack weights through lookup tables
# (more math per byte), so on a weak core they can stop being memory-bound and the bytes saved stop paying.
# One 4096 x 14336 track (the size of a real brain track) read for one word, per format, pinned to cores 4-7.
#
# Settings (environment): LG_DIR (engine folder with bin/ and lib/, default ~/preview).
set -u
DIR=${LG_DIR:-$HOME/preview}
cd "$DIR"
if pgrep -x llama-server > /dev/null || pgrep -x llama-bench > /dev/null; then
  echo "another llama process is running; stop it first"; exit 1
fi
for t in q4_0 q2_K iq2_s iq2_xs iq2_xxs iq1_m iq1_s; do
  LD_LIBRARY_PATH=$DIR/lib GGML_HEXAGON_NDEV=0 LG_N_THREADS=4 taskset -c 4-7 bin/test-backend-ops perf -o MUL_MAT -b CPU \
    -p "type_a=$t,type_b=f32,m=4096,n=1,k=14336" 2>/dev/null | grep -E "MUL_MAT" | sed "s/^ */$t  /"
done
