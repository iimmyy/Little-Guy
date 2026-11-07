#!/usr/bin/env python3
"""Everyday web text for the transplant's lessons (Immy, 2026-10-28): one C4 shard (allenai/c4, en, train 00000,
319 MB) turned into plain text. Training text from the start of the shard, held-out text from its end, so no page is
in both. HellaSwag is not in C4's sources (ActivityNet captions, WikiHow), and it is never trained on here.

usage: c4_text.py <c4 .json.gz> <out folder> [train_chars=12000000 held_chars=1200000]
"""
import gzip, json, sys

src, out = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
NT, NH = int(opt.get("train_chars", 12_000_000)), int(opt.get("held_chars", 1_200_000))
docs = [json.loads(l)["text"] for l in gzip.open(src, "rt")]
held, n = [], 0
while n < NH:
    held.append(docs.pop()); n += len(held[-1]) + 2
train, n = [], 0
for d in docs:
    if n >= NT:
        break
    train.append(d); n += len(d) + 2
open(f"{out}/train.txt", "w").write("\n\n".join(train))
open(f"{out}/held.txt", "w").write("\n\n".join(reversed(held)))
print(f"{len(docs) + len(held)} pages; train {len(train)} pages, held-out {len(held)} pages")
