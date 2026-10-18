#!/bin/bash
# Generation benchmarks (the model writes full answers that get checked) with EleutherAI's lm-evaluation-harness,
# through llama-server's chat endpoint, greedy decoding. Every sample and score is saved per brain.
#   short: GSM8K (8-shot chain of thought), IFEval, HumanEval (instruct)   server: 8 slots x 4096
#   long:  RULER needle-in-a-haystack at 4k and 16k tokens, each length run on its own with the first LIMIT_LONG
#          haystacks (the harness builds 500 per length)                    server: 2 slots x 18432
# usage: [LIMIT=n] [LIMIT_LONG=n] [TASKS_SHORT=...] [TASKS_LONG=...] gen_bench.sh <out dir> <name>=<model.gguf> [...]
set -u
B=${LG_BIN:-$HOME/work/lcpp-src/build/bin}; OUT=$1; shift; mkdir -p "$OUT"
PORT=8199; LIM=${LIMIT:+--limit $LIMIT}; LIML=${LIMIT_LONG:-100}
LM=${LG_LMEVAL:-$HOME/venv-eval/bin/lm_eval}; export HF_ALLOW_CODE_EVAL=1   # HumanEval runs the model's code
TS=${TASKS_SHORT:-gsm8k_cot,ifeval,humaneval_instruct}; TL=${TASKS_LONG:-niah_single_1,niah_single_2,niah_multikey_1}
TOK=${LG_TOKENIZER:-$HOME/work/e2b/hf}      # Gemma 4's tokenizer, local (E2B and E4B share it)

serve() {   # name model ctx slots
  if curl -s -m 2 127.0.0.1:$PORT/health >/dev/null; then echo "port $PORT busy, NOT run"; return 1; fi
  (exec $B/llama-server -m "$2" -ngl 99 -c $3 -np $4 -fa on --reasoning off --host 127.0.0.1 --port $PORT > "$OUT/$1.server-$4.log" 2>&1) &
  SPID=$!; local w=0
  until curl -s -m 2 127.0.0.1:$PORT/health | grep -q ok; do
    sleep 2; w=$((w + 2)); if [ $w -gt 600 ] || ! kill -0 $SPID 2>/dev/null; then echo "$1: server did not start"; kill $SPID 2>/dev/null; return 1; fi
  done
}
stop() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; local w=0; while curl -s -m 2 127.0.0.1:$PORT/health >/dev/null && [ $w -lt 60 ]; do sleep 1; w=$((w + 1)); done; }
evalrun() {  # name tasks slots tag extra-args
  $LM --model local-chat-completions \
    --model_args "model=$1,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=$3,max_retries=3,tokenized_requests=False,timeout=1800" \
    --tasks "$2" --apply_chat_template --fewshot_as_multiturn --gen_kwargs temperature=0 --log_samples \
    --output_path "$OUT/$1-$4" --confirm_run_unsafe_code "${@:5}" > "$OUT/$1.$4.log" 2>&1
  echo "  $1 [$2 $4]: exit $?"; grep -E "^\|" "$OUT/$1.$4.log" | grep -vE "^\|-|Tasks *\|" | head -20
}
for spec in "$@"; do
  name=${spec%%=*}; model=${spec#*=}
  echo "=== $(date +%H:%M) $name ($(basename "$model"))"
  if serve "$name" "$model" 32768 8; then evalrun "$name" "$TS" 8 short $LIM; stop; fi
  if [ -n "$TL" ] && serve "$name" "$model" 36864 2; then
    for L in 4096 16384; do
      evalrun "$name" "$TL" 2 "niah$L" --limit $LIML --metadata "{\"max_seq_lengths\":[$L],\"tokenizer\":\"$TOK\"}"
    done; stop
  fi
done
echo "GEN_BENCH_DONE"
