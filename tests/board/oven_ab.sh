#!/bin/bash
# Old oven vs new oven on the whole brain: the same prompts, greedy (temperature 0) with the drafter checking 3 guesses,
# once with the production libraries (~/llama) and once with the upgraded ones (~/oven-ultra). The text should match
# word for word (float rounding can flip a near-tie, rarely); the talking speed is printed for each.
#   MODEL=clean-Q2_K-C bash oven_ab.sh
MODEL=$HOME/models/${MODEL:-clean-Q2_K-C}.gguf
DRAFT=$HOME/models/gemma-4-E4B-assistant-masked-q8_0.gguf
PROMPTS=("Is it okay to have grapefruit with Dayvigo?" "Hey, I just got home from a long shift. How was your day?" "Explain in two sentences why the sky is blue.")
hottest() { m=0; for z in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do v=$(cat /sys/class/thermal/thermal_zone$z/temp); [ $v -gt $m ] && m=$v; done; echo $(( m / 1000 )); }

for side in old new; do
    if [ $side = old ]; then BIN=$HOME/llama/bin; LIB=$HOME/llama/lib; else BIN=$HOME/oven-ultra/bin; LIB=$HOME/oven-ultra/lib:$HOME/llama/lib; fi
    while [ $(hottest) -gt 50 ]; do sleep 5; done
    LD_LIBRARY_PATH=$LIB GGML_HEXAGON_NDEV=0 $BIN/llama-server -m $MODEL -md $DRAFT -ngl 0 -ngld 0 --device none -t 4 -td 4 \
        --spec-type draft-mtp --spec-draft-n-max 3 -c 2048 --port 8098 --host 127.0.0.1 --no-webui --reasoning off \
        > /tmp/oven_ab_$side.log 2>&1 &
    for i in $(seq 1 90); do curl -s 127.0.0.1:8098/health | grep -q ok && break; sleep 1; done
    : > /tmp/oven_ab_$side.txt
    for p in "${PROMPTS[@]}"; do
        curl -s 127.0.0.1:8098/v1/chat/completions -H 'Content-Type: application/json' \
            -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"temperature\":0,\"max_tokens\":64,\"seed\":1}" |
            python3 -c 'import json,sys; r=json.load(sys.stdin); t=r["timings"]; print(r["choices"][0]["message"]["content"].replace("\n"," / ")); print("SPEED %.2f words/s, guesses kept %d/%d" % (t["predicted_per_second"], t.get("draft_n_accepted",0), t.get("draft_n",0)), file=sys.stderr)' \
            >> /tmp/oven_ab_$side.txt 2>> /tmp/oven_ab_${side}_speed.txt
    done
    pkill -x llama-server; sleep 2
    echo "== $side oven (peak $(hottest) C)"; cat /tmp/oven_ab_${side}_speed.txt; rm -f /tmp/oven_ab_${side}_speed.txt
done
if cmp -s /tmp/oven_ab_old.txt /tmp/oven_ab_new.txt; then echo "TEXT: identical"; else echo "TEXT: differs"; diff /tmp/oven_ab_old.txt /tmp/oven_ab_new.txt; fi
