"""Layer meter: how loud is each layer's attention, FFN and PLE compared with the signal coming in?

Every layer is a chain of three inserts that each add into the word's running signal (the residual):
attention, then the FFN, then the per-layer embedding (PLE). For each insert and word, the meter reads
20*log10(|what it adds| / |signal coming in|). An insert that sits far below the signal on most words
is a 1 dB EQ: a candidate to bypass and spend its budget elsewhere.

Usage: python layer_meter.py [N_CHATS]
"""
import re, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mute_test import CHATS, DEVICE, MASTER

INSERTS = {  # insert: (module whose INPUT is the incoming signal, module whose OUTPUT is what gets added)
    "attention": ("input_layernorm", "post_attention_layernorm"),
    "FFN": ("pre_feedforward_layernorm", "post_feedforward_layernorm"),
    "PLE": ("per_layer_input_gate", "post_per_layer_input_norm"),
}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()

    readings = {}  # (layer, insert) -> list of per-word dB tensors
    incoming = {}
    layers = [(int(re.search(r"layers\.(\d+)$", name).group(1)), m) for name, m in model.named_modules()
              if type(m).__name__ == "Gemma4TextDecoderLayer"]
    for i, layer in layers:
        for insert, (src, dst) in INSERTS.items():
            def grab_in(module, inputs, key=(i, insert)):
                incoming[key] = inputs[0].float().norm(dim=-1)

            def grab_out(module, inputs, output, key=(i, insert)):
                db = 20 * torch.log10(output.float().norm(dim=-1) / incoming[key])
                readings.setdefault(key, []).append(db.flatten().cpu())

            getattr(layer, src).register_forward_hook(lambda m, inp, out, f=grab_in: f(m, inp))
            getattr(layer, dst).register_forward_hook(grab_out)
    print(f"loaded on {DEVICE} in {time.time() - t0:.0f} s, metering {len(layers)} layers", flush=True)

    chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()][:n]
    with torch.no_grad():
        for c in chats:
            model(tok(c, return_tensors="pt").input_ids[:, :1024].to(DEVICE))

    print(f"\nmedian dB each insert adds, relative to the signal coming in ({n} chats)\n")
    print(f"layer   {'attention':>10} {'FFN':>8} {'PLE':>8}")
    for i, _ in layers:
        cells = [torch.cat(readings[(i, k)]).median().item() for k in INSERTS]
        print(f"{i:5d}   " + " ".join(f"{v:+8.1f}" if k else f"{v:+10.1f}" for k, v in enumerate(cells)))
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
