"""Waveform matching (lookup decoding): how often could his next words be guessed by pasting from an earlier "chorus"?

For every word he wrote in his saved conversations, take the last few words, find where that same run happened
before, and propose the K words that followed it there. One bounce of his brain then checks the whole guess; the
words that match what he actually said are kept, plus the one word the bounce itself produces.
  context:     match against everything earlier in the same chat (persona, Immy's messages, his earlier replies)
  sample pack: match against his replies in the OTHER chats (a cross-conversation phrase cache, leave-one-out)
  both:        context first, sample pack as the fallback
Score: words per bounce (1.00 = no help; 2.00 = every bounce prints two words on average).

Usage: python lookup_hits.py
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
CHATS = ROOT / ".run/calib/chat_calib.txt"
tok = AutoTokenizer.from_pretrained(ROOT / ".run/weights/masters/e4b-qat")
MAX_N, MIN_N = 4, 2                 # match the last 4 words, then 3, then 2


def chat_tokens(chat):
    """Token ids for one chat, and a flag per token: True where it's a word he wrote."""
    ids, his = [], []
    for seg in re.split(r"(?=<\|turn>)", chat):
        if not seg:
            continue
        head, _, body = seg.partition("\n") if seg.startswith("<|turn>") else ("", "", seg)
        h = tok.encode(head + "\n", add_special_tokens=False) if head else []
        b = tok.encode(body, add_special_tokens=False)
        ids += h + b
        his += [False] * len(h) + [head == "<|turn>model"] * len(b)
    return ids, his


chats = [chat_tokens(c) for c in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if c.strip()]


def pack_from(chat_ids):
    """Sample pack: for each run of n words in his replies, the continuations that followed it, by count."""
    pack = defaultdict(Counter)
    for c in chat_ids:
        ids, his = chats[c]
        for i in range(len(ids)):
            if not his[i]:
                continue
            for n in range(MIN_N, MAX_N + 1):
                if i - n >= 0:
                    pack[tuple(ids[i - n:i])][tuple(ids[i:i + 8])] += 1
    return pack


def run(k, use_ctx, use_pack):
    words = bounces = helped = 0
    for c, (ids, his) in enumerate(chats):
        pack = pack_from([d for d in range(len(chats)) if d != c]) if use_pack else {}
        last = {}                                    # run of n words -> where it last ended, in this chat
        i = 0
        while i < len(ids):
            if not his[i]:                           # Immy's words and the persona just get read (and remembered)
                for n in range(MIN_N, MAX_N + 1):
                    if i - n >= 0:
                        last[tuple(ids[i - n:i])] = i
                i += 1
                continue
            guess = ()
            for n in range(MAX_N, MIN_N - 1, -1):
                key = tuple(ids[i - n:i])
                if use_ctx and key in last:
                    guess = tuple(ids[last[key]:last[key] + k]); break
                if use_pack and key in pack:
                    guess = pack[key].most_common(1)[0][0][:k]; break
            kept = 0
            while kept < len(guess) and i + kept < len(ids) and his[i + kept] and guess[kept] == ids[i + kept]:
                kept += 1
            step = min(kept + 1, sum(1 for j in range(i, min(i + kept + 1, len(ids))) if his[j]))
            for j in range(i, i + step):             # remember what he just said
                for n in range(MIN_N, MAX_N + 1):
                    if j - n >= 0:
                        last[tuple(ids[j - n:j])] = j
            words += step; bounces += 1; helped += kept > 0; i += step
    return words / bounces, helped / bounces, words


print(f"{len(chats)} chats, {sum(sum(h) for _, h in chats)} words he wrote")
print(f"{'guess length':>12s} {'source':>12s} | {'words/bounce':>12s} {'bounces helped':>14s}")
for k in (3, 5):
    for name, ctx, pk in (("context", True, False), ("sample pack", False, True), ("both", True, True)):
        wpb, hit, _ = run(k, ctx, pk)
        print(f"{k:12d} {name:>12s} | {wpb:12.2f} {hit:14.1%}", flush=True)
