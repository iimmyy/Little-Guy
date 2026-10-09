"""Mute test: actually silence the quiet FFN detectors (Immy's Smart Disable rule) and hear what it costs.

For each gate setting, every detector quieter than `db` below the loudest one for that word is muted,
then perplexity is measured on his conversations and on a technical write-up (commands and logs). The gate assay says how much we
could skip; this says whether skipping it is safe.

Usage: python mute_test.py [N_CHATS] [DB ...]      e.g. python mute_test.py 4 20 30 40 60
"""
import math, re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / ".run/weights/masters/e4b-qat"
CHATS = ROOT / ".run/calib/chat_calib.txt"
GENERAL = ROOT / ".run/calib/test.txt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

state = {"db": None, "muted": 0, "total": 0}


def smart_disable_forward(self, x):
    """Gemma4TextMLP.forward with Immy's rule: mute what's more than `db` below the loudest detector."""
    out = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
    if state["db"] is not None:
        level = out.abs()
        keep = level >= level.amax(dim=-1, keepdim=True) * 10 ** (-state["db"] / 20)
        state["muted"] = state["muted"] + (~keep).sum()  # stays on the GPU; .item() per layer stalls MPS
        state["total"] += keep.numel()
        out = out * keep
    return self.down_proj(out)


def perplexity(model, texts):
    total_loss, total_words = 0.0, 0
    for ids in texts:
        logits = model(ids).logits[0, :-1]
        # score in slices: a full-precision table of 262144 odds per word is ~1 GB per passage
        for s in range(0, logits.shape[0], 128):
            total_loss += F.cross_entropy(logits[s:s + 128].float(), ids[0, 1 + s:1 + s + 128], reduction="sum").item()
        total_words += ids.shape[1] - 1
        del logits
        torch.mps.empty_cache()
    return math.exp(total_loss / total_words)


def main():
    n_chats = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    settings = [None] + [int(d) for d in sys.argv[2:]] if len(sys.argv) > 2 else [None, 20, 30, 40, 60]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()
    mlp_class = next(type(m) for m in model.modules() if type(m).__name__ == "Gemma4TextMLP")
    mlp_class.forward = smart_disable_forward
    print(f"loaded on {DEVICE} in {time.time() - t0:.0f} s", flush=True)

    # Chats after the ones the assay measured, so the test listens to text the meter never saw.
    chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()][40:40 + n_chats]
    general = GENERAL.read_text()
    chunk = len(general) // n_chats
    to_ids = lambda s: tok(s, return_tensors="pt").input_ids[:, :1024].to(DEVICE)
    sets = {"his chats": [to_ids(c) for c in chats], "tech write-up": [to_ids(general[i * chunk:(i + 1) * chunk]) for i in range(n_chats)]}

    print(f"\n{'gate':>10}   {'muted':>6}   " + "   ".join(f"{k:>22}" for k in sets), flush=True)
    base = {}
    with torch.no_grad():
        for db in settings:
            state.update(db=db, muted=0, total=0)
            row = {k: perplexity(model, v) for k, v in sets.items()}
            if db is None:
                base = row
            muted = 100 * float(state["muted"]) / state["total"] if state["total"] else 0.0
            label = "off" if db is None else f"-{db} dB"
            cells = "   ".join(f"{row[k]:10.3f} ({100 * (row[k] / base[k] - 1):+5.1f}%)" for k in sets)
            print(f"{label:>10}   {muted:5.1f}%   {cells}   [{time.time() - t0:.0f} s, GPU {torch.mps.driver_allocated_memory() / 1e9:.1f} GB]", flush=True)
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
