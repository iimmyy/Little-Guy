"""Cumulative bypass: switch off the quiet tail inserts one layer at a time and listen each step.

Step 1 bypasses layer LAST, step 2 layers LAST-1..LAST, and so on down to FIRST. Each step runs three ways:
the FFN only, attention only, and both. A bypassed insert adds nothing to the running signal.

Usage: python layer_bypass.py [N_CHATS] [FIRST] [LAST]      e.g. python layer_bypass.py 4 33 39
"""
import re, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mute_test import CHATS, DEVICE, MASTER, perplexity

bypassed = set()  # (layer, "ffn" | "attn")


def main():
    n_chats = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 33
    last = int(sys.argv[3]) if len(sys.argv) > 3 else 39
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()
    for name, layer in model.named_modules():
        if type(layer).__name__ != "Gemma4TextDecoderLayer":
            continue
        i = int(re.search(r"layers\.(\d+)$", name).group(1))
        for kind, norm in (("ffn", layer.post_feedforward_layernorm), ("attn", layer.post_attention_layernorm)):
            norm.register_forward_hook(lambda m, inp, out, key=(i, kind): torch.zeros_like(out) if key in bypassed else None)
    chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()][40:40 + n_chats]
    texts = [tok(c, return_tensors="pt").input_ids[:, :1024].to(DEVICE) for c in chats]
    print(f"loaded on {DEVICE} in {time.time() - t0:.0f} s, {n_chats} unseen chats\n", flush=True)

    with torch.no_grad():
        base = perplexity(model, texts)
        print(f"{'bypassed':>14}   {'FFN only':>16}   {'attention only':>16}   {'both':>16}", flush=True)
        print(f"{'nothing':>14}   {base:8.3f} ( +0.0%)   {base:8.3f} ( +0.0%)   {base:8.3f} ( +0.0%)", flush=True)
        for start in range(last, first - 1, -1):
            layers = range(start, last + 1)
            cells = []
            for kinds in (("ffn",), ("attn",), ("ffn", "attn")):
                bypassed.clear()
                bypassed.update((l, k) for l in layers for k in kinds)
                p = perplexity(model, texts)
                cells.append(f"{p:8.3f} ({100 * (p / base - 1):+5.1f}%)")
            label = f"layer {last}" if start == last else f"layers {start}-{last}"
            print(f"{label:>14}   " + "   ".join(cells) + f"   [{time.time() - t0:.0f} s]", flush=True)
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
