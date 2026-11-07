#!/usr/bin/env python3
"""Word twins: for every piece in LFM3's word list, the E4B piece that stands for the same text (Immy, 2026-10-28:
"some of the tokens themselves can be common denominators").

A graft of E4B rooms inside LFM3 needs each room's per-layer word note, which E4B looks up by its own piece number.
LFM3 reads text in its own pieces, so each LFM3 piece needs one E4B number to look up. An exact twin (same text)
gives it directly; otherwise the E4B spelling of that text is used and its last piece taken, since that piece ends
where the LFM3 piece ends, which is the spot the note is read at.

usage: twins.py <e4b folder> <lfm3 folder> <out.pt>      writes {"e4b_id": LongTensor[V_lfm3], "exact": BoolTensor}
"""
import sys

import torch
from transformers import AutoTokenizer

a_dir, b_dir, out = sys.argv[1:4]
a = AutoTokenizer.from_pretrained(a_dir)
b = AutoTokenizer.from_pretrained(b_dir, trust_remote_code=True)
A = {}
for j in range(len(a)):
    A.setdefault(a.decode([j]), j)                      # first E4B piece with this text
V = len(b)
e4b_id, exact = torch.zeros(V, dtype=torch.long), torch.zeros(V, dtype=torch.bool)
broken = 0
for i in range(V):
    s = b.decode([i])
    if i in b.all_special_ids:                          # control pieces: map start-of-text to E4B's, the rest to pad
        e4b_id[i] = a.bos_token_id if i == b.bos_token_id else a.pad_token_id
    elif s in A:
        e4b_id[i], exact[i] = A[s], True
    else:
        if "�" in s:                               # a piece that is part of one character: no text of its own
            broken += 1
        ids = a(s, add_special_tokens=False).input_ids
        e4b_id[i] = ids[-1] if ids else a.pad_token_id
torch.save({"e4b_id": e4b_id, "exact": exact}, out)
print(f"LFM3 pieces {V}: exact twin {int(exact.sum())} ({100 * exact.float().mean():.1f}%), "
      f"last-piece stand-in {V - int(exact.sum())} (of which {broken} part-of-a-character)")
