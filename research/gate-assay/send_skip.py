"""Send skip on brain C: how much of each layer's FFN send (down projection) could be skipped per word, and what it costs.

Immy's dynamic muting, turned toward speed. A detector's loudness is only known after the gate and the up gain have
run, but once it is known, a silent detector's column of the down projection never needs to be read. That column is
the biggest single read per word (591 MB of brain C's ~2.2 GB). The rule is the same Smart Disable rule as
mute_test.py: mute every detector more than `db` below the loudest one for that word.

Runs on brain C itself: the master is loaded, then every layer matrix is overwritten with brain C's own 2-bit and 4-bit
values, read back from its GGUF. Each swap is checked against the master first; a large mismatch stops the run.
The two big lookup tables (embed_tokens, embed_tokens_per_layer) stay at master precision (Q6_K in brain C, close).

Usage:
  python send_skip.py meter [N_CHATS] [DB ...]      perplexity and skipped share per setting (default: 4 chats; 30 40 50)
  python send_skip.py behavior DB [N]               20-sample behavior checks, full brain vs muting at -DB dB
"""
import json, math, re, sys, time, types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".run/lcpp-src/gguf-py"))
try:
    import yaml  # noqa: F401  (gguf-py imports it for model cards only)
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
import gguf

sys.path.insert(0, str(ROOT / "tests/behavior"))
from mute_test import CHATS, DEVICE, GENERAL, MASTER, perplexity

BRAIN_C = ROOT / ".run/weights/experiments/clean-Q2_K-C.gguf"
SWAPS = {  # master name inside each layer -> brain C name
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output", "per_layer_input_gate": "inp_gate", "per_layer_projection": "proj",
}
WORST_DB = -6.0  # a correct swap lands well below this (2-bit is around -10 dB); near 0 dB means a mapping mistake

state = {"db": None, "muted": defaultdict(lambda: 0), "total": defaultdict(int)}


def send_skip_forward(self, x):
    """Gemma4TextMLP.forward with the Smart Disable rule, counted per layer."""
    out = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
    if state["db"] is not None:
        level = out.abs()
        keep = level >= level.amax(dim=-1, keepdim=True) * 10 ** (-state["db"] / 20)
        state["muted"][self.layer] = state["muted"][self.layer] + (~keep).sum()  # stays on the GPU until the end
        state["total"][self.layer] += keep.numel()
        out = out * keep
    return self.down_proj(out)


class DiskTable(torch.nn.Module):
    """The 5.6 GB per-layer lookup table, left in the master file and read a row at a time. Each word needs one row, so
    the table never has to sit in RAM: macOS keeps the rows it reads in its file cache and drops them under pressure
    instead of swapping (on a 24 GB Mac the full table pushed generation into 18 GB of swap)."""

    def __init__(self, tensor_name, scale):
        super().__init__()
        path = MASTER / "model.safetensors"
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        info = header[tensor_name]
        assert info["dtype"] == "BF16"
        self.rows = np.memmap(path, np.uint16, "r", offset=8 + n + info["data_offsets"][0], shape=tuple(info["shape"]))
        self.scale = torch.tensor(scale, dtype=torch.bfloat16)  # rounded to bf16, as the original module does

    def forward(self, input_ids):
        rows = torch.from_numpy(self.rows[input_ids.cpu().numpy()]).view(torch.bfloat16)
        return rows.to(input_ids.device) * self.scale.to(input_ids.device)


def load_brain_c():
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).eval()
    tensors = {t.name: t for t in gguf.GGUFReader(BRAIN_C).tensors}
    params = dict(model.named_parameters())
    worst = (-math.inf, "")
    with torch.no_grad():
        for name, p in params.items():
            m = re.match(r".*language_model\.layers\.(\d+)\.(.+)\.weight$", name)
            if not m or m.group(2) not in SWAPS:
                continue
            t = tensors[f"blk.{m.group(1)}.{SWAPS[m.group(2)]}.weight"]
            w = gguf.quants.dequantize(t.data, t.tensor_type).reshape(p.shape)
            ref = p.float().numpy()
            db = 10 * math.log10(((w - ref) ** 2).sum() / (ref ** 2).sum())
            if db > WORST_DB:
                raise SystemExit(f"{name}: brain C lands {db:.1f} dB from the master; the mapping is wrong, stopping")
            worst = max(worst, (db, name))
            p.copy_(torch.from_numpy(w))
    print(f"brain C swapped in; farthest matrix {worst[0]:.1f} dB from the master ({worst[1]})", flush=True)
    for name, mod in list(model.named_modules()):
        if name.endswith("embed_tokens_per_layer"):
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            disk = DiskTable(name + ".weight", mod.scalar_embed_scale)
            probe = torch.tensor([[2, 100, 50000, 262143]])
            assert torch.equal(mod(probe), disk(probe)), "the table read from disk differs from the loaded one"
            setattr(parent, "embed_tokens_per_layer", disk)
    del params, tensors, mod, parent  # drop the 5.6 GB in-RAM table before the brain moves to the GPU
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Gemma4TextMLP":
            mod.layer = int(re.search(r"layers\.(\d+)", name).group(1))
            type(mod).forward = send_skip_forward
    return tok, model.to(DEVICE)


def meter(n_chats, settings):
    t0 = time.time()
    tok, model = load_brain_c()
    chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()][40:40 + n_chats]
    general = GENERAL.read_text()
    chunk = len(general) // n_chats
    to_ids = lambda s: tok(s, return_tensors="pt").input_ids[:, :1024].to(DEVICE)
    sets = {"his chats": [to_ids(c) for c in chats], "tech write-up": [to_ids(general[i * chunk:(i + 1) * chunk]) for i in range(n_chats)]}
    print(f"\n{'gate':>10}   {'send skipped':>12}   " + "   ".join(f"{k:>22}" for k in sets), flush=True)
    base, per_layer = {}, {}
    with torch.no_grad():
        for db in [None] + settings:
            state.update(db=db, muted=defaultdict(lambda: 0), total=defaultdict(int))
            row = {k: perplexity(model, v) for k, v in sets.items()}
            base = base or row
            muted = sum(float(v) for v in state["muted"].values())
            total = sum(state["total"].values())
            if db is not None:
                per_layer[db] = {l: float(state["muted"][l]) / state["total"][l] for l in sorted(state["total"])}
            label = "off" if db is None else f"-{db} dB"
            cells = "   ".join(f"{row[k]:10.3f} ({100 * (row[k] / base[k] - 1):+5.1f}%)" for k in sets)
            print(f"{label:>10}   {100 * muted / total if total else 0:11.1f}%   {cells}   [{time.time() - t0:.0f} s]", flush=True)
    print("\nskipped share of the send, per layer (every 6th layer):")
    for db, layers in per_layer.items():
        print(f"  -{db} dB: " + "  ".join(f"L{l}:{100 * s:.0f}%" for l, s in layers.items() if l % 6 == 0 or l == 41))
    print(f"\ndone in {time.time() - t0:.0f} s")


def behavior(db, n):
    from common import load_persona
    from scoring import QUESTIONS, checks
    t0 = time.time()
    tok, model = load_brain_c()
    persona = load_persona()
    setups = {"brain C": None, f"brain C, -{db} dB": db}
    tally = {s: defaultdict(int) for s in setups}
    examples = {}
    with torch.no_grad():
        for key, q in QUESTIONS.items():
            ids = tok.apply_chat_template([{"role": "system", "content": persona}, {"role": "user", "content": q}],
                                          add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                          enable_thinking=False)["input_ids"].to(DEVICE)
            for setup, setting in setups.items():
                state["db"] = setting
                torch.manual_seed(1000)
                replies = []
                for b in range(0, n, 5):  # small batches keep generation in real memory on a 24 GB Mac
                    out = model.generate(ids, max_new_tokens=30, do_sample=True, temperature=0.8, top_k=64, top_p=0.95,
                                         num_return_sequences=min(5, n - b))
                    replies += [tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ") for o in out]
                    print(f"  {key} / {setup}: {len(replies)}/{n} [{time.time() - t0:.0f} s]", flush=True)
                    del out
                    torch.mps.empty_cache()
                for r in replies:
                    for check, hit in checks(key, r).items():
                        tally[setup][(key, check)] += hit
                examples[(key, setup)] = replies[:2]
    names = list(setups)
    print(f"\n{'check':30s}" + "".join(f"{s:>22s}" for s in names))
    for key in QUESTIONS:
        for check in sorted({c for (k, c) in tally[names[0]] if k == key}):
            print(f"{key + ': ' + check:30s}" + "".join(f"{tally[s][(key, check)]:>19d}/{n}" for s in names))
    print("\nexamples (first two of each):")
    for (key, setup), reps in examples.items():
        for r in reps:
            print(f"  {key:8s} {setup:>20s}:  {r[:150]}")
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "meter"
    if mode == "meter":
        meter(int(sys.argv[2]) if len(sys.argv) > 2 else 4, [int(d) for d in sys.argv[3:]] or [30, 40, 50])
    elif mode == "behavior":
        behavior(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else:
        raise SystemExit(__doc__)
