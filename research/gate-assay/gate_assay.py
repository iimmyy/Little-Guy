"""Gate assay: how often is each FFN detector's gate open during Little Guy's kind of conversation?

Gemma's FFN is gated (GeGLU): response = gelu(gate(x)) * up(x). The gate acts like an allosteric site,
scaling what the up site sends on. A detector whose gate stays closed for his conversations is a pruning
candidate; if most gates are closed on a typical word, gate-first skipping could save FFN reads.

Usage:
  python gate_assay.py peek [N]    # what gate responses look like (first N conversations, default 4)
  python gate_assay.py assay [N]   # full run with is_playing(), all conversations unless N given
"""
import re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / ".run/weights/masters/e4b-qat"
CHATS = ROOT / ".run/calib/chat_calib.txt"
OUT = ROOT / ".run/results/gate-assay"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


DB_LEVELS = (10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 96, 120)


def is_playing(output, db):
    """Immy's rule, after FL Studio's Smart Disable: a detector is audible for a word if what it sends
    (gelu(gate) * up, either polarity) is within `db` decibels of the loudest detector for that word."""
    level = output.abs()
    loudest = level.amax(dim=1, keepdim=True)
    return level >= loudest * 10 ** (-db / 20)


def conversations():
    text = CHATS.read_text()
    parts = re.split(r"(?=<\|turn>system)", text)
    return [p for p in parts if p.strip()]


class Probe:
    """Clipped onto one layer's gate_proj and up_proj; sees every word's gate and up values."""

    def __init__(self, layer, mode):
        self.layer, self.mode = layer, mode
        self.gate = None
        self.words = 0
        self.open_count = {db: None for db in DB_LEVELS}   # per detector: words where it was audible
        self.open_per_word = {db: [] for db in DB_LEVELS}  # per word: fraction of this layer's detectors audible
        self.samples = []        # peek: a random handful of gate responses and outputs

    def on_gate(self, module, inputs, output):
        self.gate = F.gelu(output[0].float(), approximate="tanh")  # gemma's gelu_pytorch_tanh

    def on_up(self, module, inputs, output):
        response, out = self.gate, self.gate * output[0].float()
        if self.mode == "peek":
            idx = torch.randint(0, response.numel(), (4096,), device=response.device)
            self.samples.append((response.flatten()[idx].cpu(), out.abs().flatten()[idx].cpu()))
        else:
            for db in DB_LEVELS:
                is_open = is_playing(out, db)
                count = is_open.sum(0).cpu().long()
                self.open_count[db] = count if self.open_count[db] is None else self.open_count[db] + count
                self.open_per_word[db].append(is_open.float().mean(1).cpu())
        self.words += response.shape[0]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "peek"
    chats = conversations()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (4 if mode == "peek" else len(chats))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()
    print(f"loaded on {DEVICE} in {time.time() - t0:.0f} s", flush=True)

    probes = []
    for name, module in model.named_modules():
        if type(module).__name__ == "Gemma4TextMLP":
            probe = Probe(int(re.search(r"layers\.(\d+)\.", name).group(1)), mode)
            module.gate_proj.register_forward_hook(probe.on_gate)
            module.up_proj.register_forward_hook(probe.on_up)
            probes.append(probe)
    print(f"probes on {len(probes)} layers, {n} of {len(chats)} conversations", flush=True)

    with torch.no_grad():
        for i, chat in enumerate(chats[:n]):
            ids = tok(chat, return_tensors="pt").input_ids.to(DEVICE)
            model(ids)
            print(f"  conversation {i + 1}/{n}: {ids.shape[1]} words", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "peek":
        pct = torch.tensor([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
        print(f"\ngate response gelu(gate) percentiles   (1% 10% 25% 50% 75% 90% 99%)   and   |output| median, 99%")
        for p in probes[::4] + [probes[-1]]:
            r = torch.cat([s[0] for s in p.samples]); o = torch.cat([s[1] for s in p.samples])
            q = torch.quantile(r, pct).tolist()
            print(f"layer {p.layer:2d}  " + " ".join(f"{v:+.3f}" for v in q) + f"   |  {o.median():.3f}  {torch.quantile(o, 0.99):.3f}")
        torch.save({p.layer: torch.cat([s[0] for s in p.samples]) for p in probes}, OUT / "peek_samples.pt")
    else:
        header = "layer " + "".join(f"{-db:7d}" for db in DB_LEVELS) + "   dB below the loudest track"
        print("\naudible per word, % of the layer's 10240 detectors")
        print(header)
        for p in probes:
            print(f"{p.layer:5d} " + "".join(f"{torch.cat(p.open_per_word[db]).mean().item() * 100:7.1f}" for db in DB_LEVELS))
        print("\nsilent on 99% of words: delete candidates, out of 10240")
        print(header)
        for p in probes:
            print(f"{p.layer:5d} " + "".join(f"{(p.open_count[db] < 0.01 * p.words).sum().item():7d}" for db in DB_LEVELS))
        torch.save({p.layer: {"words": p.words, "open_count": p.open_count,
                              "open_per_word": {db: torch.cat(v) for db, v in p.open_per_word.items()}}
                    for p in probes}, OUT / "assay.pt")
    print(f"\ndone in {time.time() - t0:.0f} s, saved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
