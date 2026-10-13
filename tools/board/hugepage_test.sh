#!/bin/bash
# Does Little Guy talk faster when his brain sits in 2 MB memory pages instead of 4 KB ones?
#
# Every memory read needs its address translated; the core caches ~1-2 thousand translations, and a 4.6 GB brain
# in 4 KB pages is ~1.1 million of them, so each word walks the page tables again and again. With 2 MB pages
# the whole brain is ~2,300. LG_HUGEPAGES=1 (fork, ggml_aligned_malloc + llama-mmap) asks the kernel for them;
# the kernel decides, so this also reports how much of the brain actually got huge pages.
#
# Four setups, each run twice in alternating order (A B C D A B C D) so heat and drift hit all of them evenly:
#   A  mmap (today)              B  mmap + LG_HUGEPAGES=1
#   C  --no-mmap                 D  --no-mmap + LG_HUGEPAGES=1
#
# Settings (environment): LG_DIR (engine folder with bin/ and lib/, default ~/preview), LG_BRAIN (brain .gguf).
set -u
DIR=${LG_DIR:-$HOME/preview}
BRAIN=${LG_BRAIN:-$HOME/models/mix-21-preview.gguf}
cd "$DIR"

if pgrep -x llama-server > /dev/null; then
  echo "a llama-server is running (the chat page?); stop it first so the test has the memory and cores to itself"
  exit 1
fi

echo "page size $(getconf PAGESIZE) bytes, kernel $(uname -r)"
for f in /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag \
         /sys/kernel/mm/transparent_hugepage/hugepages-*/enabled; do
  [ -f "$f" ] && echo "  ${f#/sys/kernel/mm/transparent_hugepage/}: $(cat "$f")"
done
grep -q CONFIG_READ_ONLY_THP_FOR_FS=y /boot/config-"$(uname -r)" 2>/dev/null && echo "  huge pages for mapped files: built in"
echo

run() {   # name, LG_HUGEPAGES value, mmap 1/0
  LD_LIBRARY_PATH=$DIR/lib GGML_HEXAGON_NDEV=0 LG_HUGEPAGES=$2 bin/llama-bench -m "$BRAIN" -ngl 0 -dev none \
    -t 4 -C 0xf0 --cpu-strict 1 -fa 1 -mmp "$3" -p 0 -n 32 -d 0,1800 -r 3 -o csv > /tmp/hp_$$.csv 2>/dev/null &
  local pid=$!
  sleep 25   # the brain is loaded and the first runs are going by now
  local huge
  huge=$(awk '/AnonHugePages|FilePmdMapped/ {kb += $2} END {printf "%.2f", kb / 1048576}' /proc/$pid/smaps_rollup 2>/dev/null)
  wait $pid
  local speeds
  speeds=$(awk -F, 'NR > 1 {gsub(/"/, ""); printf "%s ", $(NF - 1)}' /tmp/hp_$$.csv)
  printf "%-28s huge pages %5s GB   words/s at depth 0, 1800: %s\n" "$1" "${huge:-?}" "$speeds"
}

for round in 1 2; do
  echo "round $round"
  run "A mmap (today)"            0 1
  run "B mmap + hugepages"        1 1
  run "C no-mmap"                 0 0
  run "D no-mmap + hugepages"     1 0
done
rm -f /tmp/hp_$$.csv
