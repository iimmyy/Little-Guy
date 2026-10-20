#!/usr/bin/env bash
# The same made-up days lived by two brains side by side (same seeds, so the same shifts, asks and messages), with his
# live settings (no coaching sheet): the fair before-and-after for a new round. Each stream gets its own brainstem.
# usage (from the repo checkout that has brain/): tools/train/ab_days.sh OUT DAYS SEED [SEED...]
# env: A_LORA, B_LORA (the two adapters as GGUF), KEY (a brainstem key file), PERSONA (default the facts-only persona),
#      ZERO (required: his resting style, as the practice days used), REPLY_WORDS (default 600, as live), CHAT_MORE (default 10), HOURS (hard limit, default 3)
set -euo pipefail
OUT=$1; DAYS=$2; shift 2
PERSONA=${PERSONA:-persona/little_guy.v3-facts.private.txt}
mkdir -p "$OUT"
cp "$PERSONA" "$OUT/persona-used.txt"
pids=()
start() {   # name lora port
    LG_LORA=$2 LG_MOST_WORDS=700 LG_CALIBRATION=$PWD/brain/calibration.json \
        timeout "${HOURS:-3}h" python tools/mind/brainstem.py brain/brain.gguf brain/dials.npz "$KEY" "$3" \
        > "$OUT/brainstem-$1.log" 2>&1 &
    pids+=($!)
}
port=8781
for s in "$@"; do
    for side in A B; do
        lora=A_LORA; [ $side = B ] && lora=B_LORA
        start "$side-$s" "${!lora}" $port
        echo "$side-$s $port" >> "$OUT/streams.txt"
        port=$((port + 1))
    done
done
for f in "$OUT"/brainstem-*.log; do               # each brain loads before its day starts
    for i in $(seq 120); do grep -q BRAINSTEM_READY "$f" && break; sleep 5; done
    grep -q BRAINSTEM_READY "$f" || { echo "no brainstem: $f"; kill "${pids[@]}"; exit 1; }
done
days=()
while read -r name p; do
    s=${name#*-}
    BRAIN_URL=http://127.0.0.1:$p BRAIN_KEY=$KEY OWN_ACT=1 ALARM=0.5 REPLY_WORDS=${REPLY_WORDS:-600} PERSON=Immy \
        PERSONA=$PERSONA CHAT_MORE=${CHAT_MORE:-10} ZERO=$ZERO \
        timeout "${HOURS:-3}h" python tools/room/practice.py days "$DAYS" "$OUT/$name" "$s" > "$OUT/$name.txt" 2>&1 &
    days+=($!)
done < "$OUT/streams.txt"
wait "${days[@]}" || true
kill "${pids[@]}" 2>/dev/null || true                # the brainstems go when the days are done
echo "ab days done: $OUT"
