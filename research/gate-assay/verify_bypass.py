"""Verify a bypass on fresh chats, then hear his answers to the four test questions with and without it.

The candidates were picked on chats 41-44, so this scores chats it never saw (45 onward).
Usage: python verify_bypass.py [N_CHATS]
"""
import re, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/behavior"))
from common import load_persona
from mute_test import CHATS, DEVICE, MASTER, perplexity

SETUPS = {
    "nothing bypassed": set(),
    "layers 37-39 removed": {(l, k) for l in (37, 38, 39) for k in ("ffn", "attn")},
    "attention 34-39 off": {(l, "attn") for l in range(34, 40)},
    "37-39 removed + attn 34-36": {(l, k) for l in (37, 38, 39) for k in ("ffn", "attn")} | {(l, "attn") for l in (34, 35, 36)},
}
QUESTIONS = ["Is it okay to have grapefruit with Dayvigo?", "Hey Little Guy, I just got home from a long shift. How was your day?",
             "Can I leave my dog in the car for ten minutes? It's like 30 degrees out.", "What pasta should I make with garlic and spinach?"]
bypassed = set()


def main():
    n_chats = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()
    for name, layer in model.named_modules():
        if type(layer).__name__ == "Gemma4TextDecoderLayer":
            i = int(re.search(r"layers\.(\d+)$", name).group(1))
            for kind, norm in (("ffn", layer.post_feedforward_layernorm), ("attn", layer.post_attention_layernorm)):
                norm.register_forward_hook(lambda m, inp, out, key=(i, kind): torch.zeros_like(out) if key in bypassed else None)
    chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()][44:44 + n_chats]
    texts = [tok(c, return_tensors="pt").input_ids[:, :1024].to(DEVICE) for c in chats]
    print(f"loaded on {DEVICE} in {time.time() - t0:.0f} s, {len(texts)} fresh chats\n", flush=True)

    with torch.no_grad():
        base = None
        for label, setup in SETUPS.items():
            bypassed.clear(); bypassed.update(setup)
            p = perplexity(model, texts)
            base = base or p
            print(f"{label:>28}   {p:8.3f} ({100 * (p / base - 1):+5.1f}%)   [{time.time() - t0:.0f} s]", flush=True)

        persona = load_persona()
        for q in QUESTIONS:
            print(f"\nQ: {q}", flush=True)
            ids = tok.apply_chat_template([{"role": "system", "content": persona}, {"role": "user", "content": q}],
                                          add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                          enable_thinking=False)["input_ids"].to(DEVICE)
            for label in ("nothing bypassed", "layers 37-39 removed"):
                bypassed.clear(); bypassed.update(SETUPS[label])
                out = model.generate(ids, max_new_tokens=60, do_sample=False)
                reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
                print(f"  {label:>20}:  {reply}", flush=True)
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
