"""How big is the dictionary for English + German + Russian + Korean, cut by alphabet?"""
import sys, unicodedata
from collections import Counter
sys.path.insert(0, "/Users/immyyyyy/Documents/LittleGuy/.run/lcpp-src/gguf-py")
from gguf import GGUFReader

r = GGUFReader("/Users/immyyyyy/Documents/LittleGuy/.run/models/gemma-4-E4B_q4_0-it.gguf")
f_tok, f_type = r.fields["tokenizer.ggml.tokens"], r.fields["tokenizer.ggml.token_type"]
pieces = [bytes(f_tok.parts[i]).decode("utf-8", "replace") for i in f_tok.data]
types = [int(f_type.parts[i][0]) for i in f_type.data]


def script(ch):
    if ch.isspace() or ch == "▁" or not ch.isalpha():
        return "common"  # digits, punctuation, symbols, spaces
    name = unicodedata.name(ch, "")
    for s in ("LATIN", "CYRILLIC", "HANGUL", "CJK", "ARABIC", "DEVANAGARI", "GREEK", "HEBREW", "THAI", "HIRAGANA", "KATAKANA"):
        if name.startswith(s) or f" {s} " in f" {name} ":
            return s
    return "other"


counts, keep = Counter(), []
for i, (p, t) in enumerate(zip(pieces, types)):
    if t in (3, 4, 6) or len(p.replace("▁", "")) <= 1:   # control/byte entries and single characters: always keep
        counts["safety net"] += 1; keep.append(i); continue
    scripts = {script(c) for c in p.replace("▁", " ")} - {"common"}
    label = "common only" if not scripts else scripts.pop() if len(scripts) == 1 else "mixed"
    counts[label] += 1
    if label in ("common only", "LATIN", "CYRILLIC", "HANGUL"):
        keep.append(i)
for k, v in counts.most_common():
    print(f"  {k:14s} {v:7d}")
print(f"keep (EN/DE/RU/KO + common + safety): {len(keep)} of {len(pieces)} = {len(keep) / len(pieces):.1%}")
open("/Users/immyyyyy/Documents/LittleGuy/.run/trim/keep_scripts.txt", "w").write("\n".join(map(str, keep)))
