"""20 samples per question for each bypass setup, scored with the shared behavior checks.

Usage: python bypass_behavior.py [N]
"""
import re, sys, time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/behavior"))
from common import load_persona
from mute_test import DEVICE, MASTER
from scoring import QUESTIONS, checks

SETUPS = {
    "full brain": set(),
    "37-39 removed": {(l, k) for l in (37, 38, 39) for k in ("ffn", "attn")},
}
bypassed = set()
BATCH = 5
MAX_WORDS = 30  # every check reads the first sentence or a short reply


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()
    for name, layer in model.named_modules():
        if type(layer).__name__ == "Gemma4TextDecoderLayer":
            i = int(re.search(r"layers\.(\d+)$", name).group(1))
            for kind, norm in (("ffn", layer.post_feedforward_layernorm), ("attn", layer.post_attention_layernorm)):
                norm.register_forward_hook(lambda m, inp, out, key=(i, kind): torch.zeros_like(out) if key in bypassed else None)
    persona = load_persona()
    print(f"loaded on {DEVICE} in {time.time() - t0:.0f} s; {n} samples per question at temperature 0.8\n", flush=True)

    tally = {s: defaultdict(int) for s in SETUPS}
    examples = {}
    with torch.no_grad():
        for key, q in QUESTIONS.items():
            ids = tok.apply_chat_template([{"role": "system", "content": persona}, {"role": "user", "content": q}],
                                          add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                          enable_thinking=False)["input_ids"].to(DEVICE)
            for setup, parts in SETUPS.items():
                bypassed.clear(); bypassed.update(parts)
                torch.manual_seed(1000)
                replies = []
                for b in range(0, n, BATCH):  # small batches keep generation in real memory on a 24 GB Mac
                    out = model.generate(ids, max_new_tokens=MAX_WORDS, do_sample=True, temperature=0.8, top_k=64, top_p=0.95,
                                         num_return_sequences=min(BATCH, n - b))
                    replies += [tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ") for o in out]
                    print(f"  {key} / {setup}: {len(replies)}/{n} [{time.time() - t0:.0f} s]", flush=True)
                    del out
                    torch.mps.empty_cache()
                for r in replies:
                    for check, hit in checks(key, r).items():
                        tally[setup][(key, check)] += hit
                examples[(key, setup)] = replies[:2]

    names = list(SETUPS)
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
    main()
