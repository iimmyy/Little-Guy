"""Hear the mute test: the same question with the Smart Disable gate at different settings.

Usage: python mute_demo.py "question" [DB ...]      e.g. python mute_demo.py "How was your day?" 20 30 40 60
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/behavior"))
from common import load_persona
from mute_test import DEVICE, MASTER, smart_disable_forward, state


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "Hey Little Guy, how was your day?"
    settings = [None] + [int(d) for d in sys.argv[2:]] if len(sys.argv) > 2 else [None, 20, 30, 40, 60]
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()
    next(type(m) for m in model.modules() if type(m).__name__ == "Gemma4TextMLP").forward = smart_disable_forward
    messages = [{"role": "system", "content": load_persona()}, {"role": "user", "content": question}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                  enable_thinking=False)["input_ids"].to(DEVICE)
    print(f"Q: {question}\n", flush=True)
    with torch.no_grad():
        for db in settings:
            state.update(db=db, muted=0, total=0)
            out = model.generate(ids, max_new_tokens=50, do_sample=False)
            reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
            print(f"{'off' if db is None else f'-{db} dB':>7}:  {reply}\n", flush=True)


if __name__ == "__main__":
    main()
