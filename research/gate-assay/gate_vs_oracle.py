"""Can the gate alone decide what to mute? A blind A/B at the same mute amount.

For each word, mute the same fraction of FFN detectors two ways:
  oracle: pick the quietest by what each detector actually sends, gelu(gate) * up (needs everything computed)
  gate:   pick the quietest by the gate alone, |gelu(gate)| (could be decided before computing up and down)
If the gate's perplexity lands near the oracle's, real-time skipping on the board is possible.

Usage: python gate_vs_oracle.py [N_CHATS] [MUTE_PERCENT ...]   e.g. python gate_vs_oracle.py 4 50 62 75 85
"""
import re, sys, time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mute_test import CHATS, DEVICE, MASTER, perplexity

state = {"rule": None, "keep": 1.0}


def ab_forward(self, x):
    g = F.gelu(self.gate_proj(x), approximate="tanh")
    out = g * self.up_proj(x)
    if state["rule"] is not None:
        score = (out if state["rule"] == "oracle" else g).abs()
        k = max(1, round(state["keep"] * score.shape[-1]))
        cutoff = score.topk(k, dim=-1).values[..., -1:]  # the k-th loudest per word
        out = out * (score >= cutoff)
    return self.down_proj(out)


def main():
    n_chats = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    mutes = [int(m) for m in sys.argv[2:]] or [50, 62, 75, 85]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()
    next(type(m) for m in model.modules() if type(m).__name__ == "Gemma4TextMLP").forward = ab_forward
    chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()][40:40 + n_chats]
    texts = [tok(c, return_tensors="pt").input_ids[:, :1024].to(DEVICE) for c in chats]
    print(f"loaded on {DEVICE} in {time.time() - t0:.0f} s, {n_chats} unseen chats\n", flush=True)

    with torch.no_grad():
        state.update(rule=None)
        base = perplexity(model, texts)
        print(f"{'muted':>7}   {'oracle (bounce)':>18}   {'gate only (real time)':>22}", flush=True)
        print(f"{'0%':>7}   {base:10.3f} ( +0.0%)   {base:13.3f} ( +0.0%)", flush=True)
        for m in mutes:
            row = []
            for rule in ("oracle", "gate"):
                state.update(rule=rule, keep=1 - m / 100)
                p = perplexity(model, texts)
                row.append(f"{p:10.3f} ({100 * (p / base - 1):+5.1f}%)")
            print(f"{m:>6}%   {row[0]:>18}   {row[1]:>22}   [{time.time() - t0:.0f} s]", flush=True)
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
