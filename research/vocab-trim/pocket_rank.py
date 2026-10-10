"""Pocket dictionary by commonness: keep the safety net (single characters, bytes, control entries) plus the
most common entries written in his five languages' alphabets, ranked by when the tokenizer learned them
(BPE merge order: common pieces are glued first). Graded on his real hidden states in EN/DE/RU/KO/JA:
how much of his word-choice probability lands inside the pocket, and how often his #1 word is in it.
"""
import sys, unicodedata
import numpy as np

from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
sys.path.insert(0, f"{RUN}/lcpp-src/gguf-py")
from gguf import GGUFReader

DATA = f"{RUN}/results/vocab-trim"
CAP = 30.0                                                     # Gemma's final logit softcap
r = GGUFReader(f"{RUN}/weights/brains/gemma-4-E4B_q4_0-it.gguf")
f_tok, f_type, f_mrg = (r.fields[f"tokenizer.ggml.{k}"] for k in ("tokens", "token_type", "merges"))
pieces = [bytes(f_tok.parts[i]).decode("utf-8", "replace") for i in f_tok.data]
types = [int(f_type.parts[i][0]) for i in f_type.data]
index = {p: i for i, p in enumerate(pieces)}

rank = np.full(len(pieces), 10**9, dtype=np.int64)            # merge rank of the entry each merge creates
for m, i in enumerate(f_mrg.data):
    a, _, b = bytes(f_mrg.parts[i]).decode("utf-8", "replace").partition(" ")
    t = index.get(a + b)
    if t is not None and rank[t] > m:
        rank[t] = m

ALPHABETS = ("LATIN", "CYRILLIC", "HANGUL", "CJK", "HIRAGANA", "KATAKANA")


def ours(p):
    """True if every letter in the entry belongs to one of his languages' alphabets (digits/punctuation are fine)."""
    for c in p.replace("▁", " "):
        if c.isalpha() and not unicodedata.name(c, "").startswith(ALPHABETS):
            return False
    return True


safety = [i for i, (p, t) in enumerate(zip(pieces, types)) if t in (3, 4, 6) or len(p.replace("▁", "")) <= 1]
safe = set(safety)
candidates = [i for i in range(len(pieces)) if i not in safe and ours(pieces[i])]
candidates.sort(key=lambda i: (rank[i], i))
print(f"full dictionary {len(pieces)} | safety net {len(safety)} | in his alphabets {len(candidates)} "
      f"| with a merge rank {int((rank[candidates] < 10**9).sum())}")

E = np.load(f"{DATA}/E.npy")
H = dict(np.load(f"{DATA}/H.npz"))
probs = {}
for lang, h in H.items():                                      # his full word-choice odds at every position
    z = CAP * np.tanh((h @ E.T) / CAP)
    p = np.exp(z - z.max(1, keepdims=True)).astype(np.float32)
    probs[lang] = p / p.sum(1, keepdims=True)

print(f"{'pocket size':>11s} {'share':>6s} | " + "  ".join(f"{l[:3]:>11s}" for l in H) + " | worst spot")
print(f"{'':>11s} {'':>6s} | " + "  ".join(f"{'mass  #1in':>11s}" for _ in H) + " |")
for n in (32768, 49152, 65536, 98304, 131072):
    keep = np.zeros(len(pieces), bool)
    keep[safety] = True
    keep[candidates[:max(0, n - len(safety))]] = True
    cells, worst = [], 1.0
    for lang, p in probs.items():
        mass = p[:, keep].sum(1)
        top1 = keep[p.argmax(1)].mean()
        cells.append(f"{mass.mean():5.1%} {top1:5.1%}")
        worst = min(worst, mass.min())
    print(f"{int(keep.sum()):11d} {keep.mean():6.1%} | " + "  ".join(cells) + f" | {worst:6.1%}")
