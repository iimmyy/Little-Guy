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
  python send_skip.py perlayer [N_CHATS] [BUDGET]   per-layer gate search aimed at board speed: each block of 6 layers
                                                    at every gate level (or none), then the combination that saves
                                                    the most board send time within BUDGET percent chat cost (default 3)
  python send_skip.py greedy [N_CHATS] [BUDGET]     the same search, tuned step by step on the real combination
                                                    (per-block costs don't add up), checked on 8 unseen chats
DB can also be a per-layer list, "0,0,35,..." (42 values, 0 = no gate), for behavior.
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
import os
from mute_test import CHATS, DEVICE, GENERAL, MASTER, perplexity

# the cloud job (tools/azure/jobs/ml-gate-sweep.yaml) points these at its mounted copies and at public text;
# his conversations never leave the Mac, so off the Mac the search tunes on SEND_SKIP_TEXT
MASTER = Path(os.environ.get("SEND_SKIP_MASTER", MASTER))
BRAIN_C = Path(os.environ.get("SEND_SKIP_BRAIN", ROOT / ".run/weights/experiments/clean-Q2_K-C.gguf"))
if DEVICE != "mps":
    torch.mps.empty_cache = lambda: None   # perplexity() frees Mac GPU memory between passages


def passages():
    """His chats on the Mac; with SEND_SKIP_TEXT set, 4,000-character passages of that text file instead."""
    if os.environ.get("SEND_SKIP_TEXT"):
        text = Path(os.environ["SEND_SKIP_TEXT"]).read_text()
        return [text[i:i + 4000] for i in range(0, len(text) - 4000, 4000)]
    return [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()]
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
    db = state["db"][self.layer] if isinstance(state["db"], list) else state["db"]
    if db:   # None or 0: this layer is not gated
        level = out.abs()
        keep = level >= level.amax(dim=-1, keepdim=True) * 10 ** (-db / 20)
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


# Board measurement (tests/oven/send_skip_kernel.c: the skip kernel with its real arithmetic against the oven's Q4_0
# gemv, 4 fast cores): a layer's send time when only its open detectors are read, as a share of today's send, by how
# silent the layer is. It breaks even near 53% silent; below that a layer keeps today's recipe (send_time caps at 1).
BOARD_CURVE = [(0.0, 1.95), (0.5, 1.062), (0.6, 0.894), (0.7, 0.693), (0.8, 0.48), (0.85, 0.358), (0.9, 0.246),
               (0.95, 0.129), (0.98, 0.06), (1.0, 0.0)]
LEVELS = (None, 50, 45, 40, 35, 30, 25)   # None: no gate in that block (read every row)


def send_time(silent):
    """Board time for one layer's send at this silent share, as a share of today's send (never above 1: a layer
    where skipping would be slower keeps today's recipe)."""
    for (s0, t0), (s1, t1) in zip(BOARD_CURVE, BOARD_CURVE[1:]):
        if silent <= s1:
            return min(1.0, t0 + (silent - s0) / (s1 - s0) * (t1 - t0))
    return 0.0


def perlayer(n_chats, budget):
    """Measure every block of 6 layers at every gate level (the rest ungated), then pick per block the level that
    saves the most board time while the summed chat cost stays within budget, and check that combination."""
    t0 = time.time()
    tok, model = load_brain_c()
    chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()][40:40 + n_chats]
    texts = [tok(c, return_tensors="pt").input_ids[:, :1024].to(DEVICE) for c in chats]

    def measure(setting):
        state.update(db=setting, muted=defaultdict(lambda: 0), total=defaultdict(int))
        with torch.no_grad():
            ppl = perplexity(model, texts)
        return ppl, {l: float(state["muted"][l]) / state["total"][l] for l in state["total"]}

    full, _ = measure(None)
    cost = lambda p: 100 * (p / full - 1)
    blocks = [list(range(b, min(b + 6, 42))) for b in range(0, 42, 6)]
    table = {}   # (block, level) -> (cost, {layer: silent share})
    for bi, block in enumerate(blocks):
        table[(bi, None)] = (0.0, {l: 0.0 for l in block})
        for level in LEVELS[1:]:
            p, silent = measure([level if l in block else None for l in range(42)])
            table[(bi, level)] = (cost(p), {l: silent[l] for l in block})
            saved = sum(1 - send_time(silent[l]) for l in block) / 42
            print(f"  layers {block[0]:2d}-{block[-1]:2d} at -{level} dB: silent {100 * sum(silent[l] for l in block) / len(block):4.1f}%, "
                  f"send time saved {100 * saved:4.1f} pts, chat cost {cost(p):+.2f}%  [{time.time() - t0:.0f} s]", flush=True)
    json.dump({f"{b},{lv}": v for (b, lv), v in table.items()}, open(ROOT / ".run/results/send_skip_perlayer.json", "w"), indent=1)
    plan(budget, tok, model)


def plan(budget, tok=None, model=None):
    """Pick per block the gate level that saves the most board send time within the chat-cost budget (from the saved
    per-block measurements), then check the combination on the 4 tuning chats and 8 chats the search never saw."""
    t0 = time.time()
    raw = json.load(open(ROOT / ".run/results/send_skip_perlayer.json"))
    table = {}
    for key, (c, silent) in raw.items():
        b, lv = key.split(",")
        table[(int(b), None if lv == "None" else int(lv))] = (c, {int(l): s for l, s in silent.items()})
    n_blocks = 1 + max(b for b, _ in table)
    best = (-1.0, None)
    for combo in __import__("itertools").product(LEVELS, repeat=n_blocks):
        spent = sum(max(table[(b, lv)][0], 0.0) for b, lv in enumerate(combo))   # a block that "helps" counts as free
        if spent <= budget:
            saved = sum(1 - send_time(s) for b, lv in enumerate(combo) for s in table[(b, lv)][1].values()) / 42
            if saved > best[0]:
                best = (saved, combo)
    saved, combo = best
    setting = [combo[l // 6] for l in range(42)]
    print("chosen: " + "  ".join(f"layers {6 * b}-{min(6 * b + 5, 41)}: {'off' if lv is None else f'-{lv} dB'}"
                                 for b, lv in enumerate(combo)))
    print(f"estimate: send time saved {100 * saved:.1f}%")
    if model is None:
        tok, model = load_brain_c()
    all_chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()]
    for name, chats in (("tuning chats (40-43)", all_chats[40:44]), ("unseen chats (44-51)", all_chats[44:52])):
        texts = [tok(c, return_tensors="pt").input_ids[:, :1024].to(DEVICE) for c in chats]
        with torch.no_grad():
            state.update(db=None)
            full = perplexity(model, texts)
            state.update(db=setting, muted=defaultdict(lambda: 0), total=defaultdict(int))
            gated = perplexity(model, texts)
        silent = {l: float(state["muted"][l]) / state["total"][l] for l in state["total"]}
        real = sum(1 - send_time(silent.get(l, 0.0)) for l in range(42)) / 42
        print(f"{name}: send time saved {100 * real:.1f}%, chat cost {100 * (gated / full - 1):+.2f}%  [{time.time() - t0:.0f} s]", flush=True)
    print("setting for behavior: " + ",".join("0" if lv is None else str(lv) for lv in setting))


PROBE = ROOT / ".run/results/send_skip_probe_dayvigo.json"


def warn_probe(tok, model):
    """A fast safety probe for the gate search (perplexity misses it: gates that cost under 1% perplexity cut his
    Dayvigo warn-first from 28/40 to 12/40). Full brain C answers the Dayvigo question 40 times once (cached); for any
    setting, the probe scores those replies teacher-forced and returns how much more likely, per word, the brain finds
    its warn-first replies than its other replies. A gate that lowers this is pushing him away from the warning."""
    from common import load_persona
    from scoring import QUESTIONS, checks
    ids = tok.apply_chat_template([{"role": "system", "content": load_persona()}, {"role": "user", "content": QUESTIONS["dayvigo"]}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                  enable_thinking=False)["input_ids"].to(DEVICE)
    if not PROBE.exists():
        state["db"] = None
        replies = []
        with torch.no_grad():
            for seed in (1000, 7000):
                torch.manual_seed(seed)
                for b in range(0, 20, 5):
                    out = model.generate(ids, max_new_tokens=30, do_sample=True, temperature=0.8, top_k=64, top_p=0.95, num_return_sequences=5)
                    replies += [tok.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]
                    torch.mps.empty_cache()
        json.dump([{"reply": r, "warns": checks("dayvigo", r)["warns FIRST"]} for r in replies], open(PROBE, "w"), indent=1)
    sets = json.load(open(PROBE))

    def opening(reply):   # the reply up to the end of its first spoken sentence (where warn or dodge is decided)
        m = re.search(r"[.!?](\s|$)", re.sub(r"\[[^\]]*\]", lambda t: " " * len(t.group()), reply).lstrip() and
                      re.sub(r"\[[^\]]*\]", lambda t: " " * len(t.group()), reply))
        return reply[:m.end()] if m else reply
    encoded = [(tok(opening(r["reply"]), add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE), r["warns"]) for r in sets]

    def logp_openings():
        out = []
        with torch.no_grad():
            for r, _ in encoded:
                logits = model(torch.cat([ids, r], dim=1)).logits[0, ids.shape[1] - 1:-1].float()
                out.append(F.log_softmax(logits, -1).gather(1, r[0][:, None]).sum().item())
        return out
    saved = state["db"]
    state["db"] = None
    base = logp_openings()   # the full brain, which sampled these replies
    state["db"] = saved

    def score():
        """Estimated warn-first rate under the current setting: the full brain's replies, each weighted by how much
        more (or less) likely the gated brain finds its opening (importance weighting; 28/40 for the full brain)."""
        w = [math.exp(g - b) for g, b in zip(logp_openings(), base)]
        return sum(wi for wi, (_, warns) in zip(w, encoded) if warns) / sum(w)
    return score


def greedy(n_chats, budget):
    """Tune on the real combination: from every block ungated, repeatedly try one notch tighter on each block on top of
    what's already chosen, keep the move that saves the most board send time per point of chat cost, and stop when
    every next move would cross the budget. Then check the result on 8 chats the tuning never saw."""
    t0 = time.time()
    tok, model = load_brain_c()
    all_chats = passages()
    n_tok = int(os.environ.get("SEND_SKIP_TOKENS", "1024"))   # shorter passages keep the CPU job to a few hours
    tune = [tok(c, return_tensors="pt").input_ids[:, :n_tok].to(DEVICE) for c in all_chats[40:40 + n_chats]]
    unseen = [tok(c, return_tensors="pt").input_ids[:, :n_tok].to(DEVICE) for c in all_chats[44:52]]

    def measure(setting, texts):
        state.update(db=setting, muted=defaultdict(lambda: 0), total=defaultdict(int))
        with torch.no_grad():
            ppl = perplexity(model, texts)
        silent = {l: float(state["muted"][l]) / state["total"][l] for l in state["total"]}
        return ppl, sum(1 - send_time(silent.get(l, 0.0)) for l in range(42)) / 42

    full, _ = measure(None, tune)
    table = json.load(open(ROOT / ".run/results/send_skip_perlayer.json")) if (ROOT / ".run/results/send_skip_perlayer.json").exists() else {}
    worthless = lambda b, lv: f"{b},{lv}" in table and all(send_time(x) >= 1 for x in table[f"{b},{lv}"][1].values())
    combo, cost_now, saved_now = [None] * 7, 0.0, 0.0
    while True:
        moves = []
        for b, lv in ((b, lv) for b in range(7) for lv in LEVELS[LEVELS.index(combo[b]) + 1:]):
            # a block saves nothing until skipping beats today's recipe (about 53% silent on the board), so a move
            # may jump several notches; levels the per-block table says stay below break-even are not tried
            if worthless(b, lv):
                continue
            trial = combo[:b] + [lv] + combo[b + 1:]
            ppl, saved = measure([trial[l // 6] for l in range(42)], tune)
            cost = 100 * (ppl / full - 1)
            if cost <= budget and saved > saved_now:
                moves.append(((saved - saved_now) / max(cost - cost_now, 0.25), trial, cost, saved))
        if not moves:
            break
        _, combo, cost_now, saved_now = max(moves, key=lambda m: m[0])
        print("  " + " ".join("--" if lv is None else str(lv) for lv in combo) +
              f"   send time saved {100 * saved_now:4.1f}%, chat cost {cost_now:+.2f}%  [{time.time() - t0:.0f} s]", flush=True)
    setting = [combo[l // 6] for l in range(42)]
    print("\nchosen: " + "  ".join(f"layers {6 * b}-{min(6 * b + 5, 41)}: {'off' if lv is None else f'-{lv} dB'}" for b, lv in enumerate(combo)))
    full_u, _ = measure(None, unseen)
    ppl_u, saved_u = measure(setting, unseen)
    print(f"unseen chats (44-51): send time saved {100 * saved_u:.1f}%, chat cost {100 * (ppl_u / full_u - 1):+.2f}%")
    print("setting for behavior: " + ",".join("0" if lv is None else str(lv) for lv in setting))
    if os.environ.get("SEND_SKIP_OUT"):
        json.dump({"chosen": [lv for lv in combo], "unseen_send_time_saved": saved_u,
                   "unseen_cost_pct": 100 * (ppl_u / full_u - 1), "text": os.environ.get("SEND_SKIP_TEXT", "his chats")},
                  open(os.environ["SEND_SKIP_OUT"], "w"), indent=1)
    print(f"\ndone in {time.time() - t0:.0f} s")


def behavior(db, n):
    from common import load_persona
    from scoring import QUESTIONS, checks
    t0 = time.time()
    tok, model = load_brain_c()
    persona = load_persona()
    label = f"-{db} dB" if isinstance(db, int) else "per-layer gate"
    setups = {"brain C": None, f"brain C, {label}": db}
    tally = {s: defaultdict(int) for s in setups}
    examples = {}
    with torch.no_grad():
        for key, q in QUESTIONS.items():
            ids = tok.apply_chat_template([{"role": "system", "content": persona}, {"role": "user", "content": q}],
                                          add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                          enable_thinking=False)["input_ids"].to(DEVICE)
            for setup, setting in setups.items():
                state["db"] = setting
                torch.manual_seed(int(__import__("os").environ.get("SEND_SKIP_SEED", "1000")))   # fresh dice: another seed
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
        arg = sys.argv[2]
        behavior([int(x) for x in arg.split(",")] if "," in arg else int(arg), int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    elif mode == "greedy":
        greedy(int(sys.argv[2]) if len(sys.argv) > 2 else 4, float(sys.argv[3]) if len(sys.argv) > 3 else 3.0)
    elif mode == "probe":   # calibrate the warn probe on settings with known 40-try warn-first counts
        tok, model = load_brain_c()
        score = warn_probe(tok, model)
        known = {"full brain C (28/40)": None, "uniform -45 dB (13/20 on one seed)": 45,
                 "first greedy (24/40)": [b for b in (35, 50, 40, 50, 45, 30, 35) for _ in range(6)],
                 "setting A (18/40)": [b for b in (35, 45, 0, 0, 30, 30, 35) for _ in range(6)],
                 "setting B (12/40)": [b for b in (40, 45, 0, 0, 30, 30, 35) for _ in range(6)]}
        for name, setting in known.items():
            state.update(db=setting)
            print(f"{name:36s} estimated warn-first {40 * score():5.1f}/40", flush=True)
    elif mode == "plan":
        plan(float(sys.argv[2]) if len(sys.argv) > 2 else 3.0)
    elif mode == "perlayer":
        perlayer(int(sys.argv[2]) if len(sys.argv) > 2 else 4, float(sys.argv[3]) if len(sys.argv) > 3 else 3.0)
    else:
        raise SystemExit(__doc__)
