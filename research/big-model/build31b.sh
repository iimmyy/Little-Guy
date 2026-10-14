#!/bin/bash
# Gemma 4 31B through Little Guy's recipe (any Linux box with 16+ cores, 64 GB RAM, ~200 GB disk at $S). Steps skip themselves when their output exists.
set -euo pipefail
S=${S:-/scratch}; B=$S/src/build/bin; . $S/venv/bin/activate
MASTER_REV=1e4d8beecacb8b7590c1d8bedd7335f687bf311f   # google/gemma-4-31B-it-qat-q4_0-unquantized, 2026-07-20
# $S/kl_text.txt is docs/2026-09-28.md from this repo; $S/add_dict_preview.py is tools/quantize/add_dict_preview.py
log() { echo "== $(date -u +%H:%M:%S) $*"; }
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -f $S/master/model-00002-of-00002.safetensors ] || { log download; hf download google/gemma-4-31B-it-qat-q4_0-unquantized --revision $MASTER_REV --local-dir $S/master; }
[ -f $S/31b-bf16.gguf ] || { log convert; python $S/convert_wrap.py $S/src $S/master --outtype bf16 --outfile $S/31b-bf16.gguf; }
[ -f $S/31b-q8_0.gguf ] || { log q8_0; $B/llama-quantize $S/31b-bf16.gguf $S/31b-q8_0.gguf Q8_0 16; }
# brain C recipe: careful 2-bit gate/up/q/k, 4-bit down/out/v, 6-bit dictionary, no importance matrix
[ -f $S/31b-C.gguf ] || { log "C (careful 2-bit)"; LG_Q2K_CAREFUL=1 $B/llama-quantize \
    --tensor-type ffn_gate=q2_k --tensor-type ffn_up=q2_k --tensor-type attn_q=q2_k --tensor-type attn_k=q2_k \
    --tensor-type ffn_down=q4_0 --tensor-type attn_output=q4_0 --tensor-type attn_v=q4_0 \
    --token-embedding-type q6_k $S/31b-bf16.gguf $S/31b-C.gguf Q4_0 16; }
[ -f $S/31b-rough.gguf ] || { log rough; $B/llama-quantize --allow-requantize --tensor-type token_embd=q2_k --token-embedding-type q2_k $S/31b-C.gguf $S/31b-rough.gguf Q2_K 16; }
[ -f $S/31b-C-preview.gguf ] || { log preview; python $S/add_dict_preview.py $S/src/gguf-py $S/31b-C.gguf $S/31b-rough.gguf $S/31b-C-preview.gguf; rm -f $S/31b-rough.gguf; }
[ -f $S/31b_kl_base.bin ] || { log "kl base (q8_0)"; LG_DICT_PREVIEW=0 $B/llama-perplexity -m $S/31b-q8_0.gguf -f $S/kl_text.txt -c 512 --chunks 4 -t 16 --kl-divergence-base $S/31b_kl_base.bin > $S/kl_base.log 2>&1; }
log "kl C vs q8_0"; LG_DICT_PREVIEW=0 $B/llama-perplexity -m $S/31b-C-preview.gguf -f $S/kl_text.txt -c 512 --chunks 4 -t 16 --kl-divergence-base $S/31b_kl_base.bin --kl-divergence 2>&1 | grep -E "Mean    KLD|Same top|Mean PPL" || true
log done; ls -la $S/*.gguf
