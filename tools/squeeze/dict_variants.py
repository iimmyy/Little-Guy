#!/usr/bin/env python3
"""Immy's idea: store near-twin tokens ("hello", " hello", "Hello") as one base plus a small difference.

For each variant kind (leading space, capital first letter), over every token pair in the vocab that differs only by
that change: how big is the raw difference between the two dictionary rows, and how big is what's left after removing
one shared push for that kind (the average difference)? Small leftovers = the variants can be stored as base + shared
push + a few-bit leftover, and scored with one shared dot product per kind.
usage: dict_variants.py <bf16 gguf> <tokenizer.json>
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py"))
import numpy as np
import gguf

T = gguf.GGMLQuantizationType
t = next(x for x in gguf.GGUFReader(sys.argv[1]).tensors if x.name == "token_embd.weight")
E = (np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(int(t.shape[1]), -1)
vocab = json.load(open(sys.argv[2]))["model"]["vocab"]
tok = vocab if isinstance(vocab, dict) else {p[0]: i for i, p in enumerate(vocab)}
print(f"dictionary {E.shape[0]} x {E.shape[1]}, {len(tok)} token strings")

SP = "▁"                                         # SentencePiece's leading-space mark
words = {s for s in tok if s.lstrip(SP).isalpha()}
core = {}
for s in words:
    core.setdefault(s.lstrip(SP).lower(), []).append(s)
grouped = sum(len(v) for v in core.values() if len(v) > 1)
print(f"letter-only tokens {len(words)}; in a family of 2+ spellings (space / case variants): {grouped} "
      f"({grouped / E.shape[0]:.1%} of the dictionary) in {sum(1 for v in core.values() if len(v) > 1)} families")

rng = np.random.default_rng(0)
norm = np.linalg.norm(E, axis=1)


def kind(name, pairs):
    a = np.array([tok[x] for x, _ in pairs]); b = np.array([tok[y] for _, y in pairs])
    d = E[b] - E[a]
    push = d.mean(0)
    raw = np.linalg.norm(d, axis=1) / norm[b]
    left = np.linalg.norm(d - push, axis=1) / norm[b]
    cos = (E[a] * E[b]).sum(1) / (norm[a] * norm[b])
    r = rng.integers(0, E.shape[0], (len(pairs), 2))
    rcos = (E[r[:, 0]] * E[r[:, 1]]).sum(1) / (norm[r[:, 0]] * norm[r[:, 1]])
    print(f"{name}: {len(pairs)} pairs")
    print(f"  similarity of the pair (cosine, 1 = same): median {np.median(cos):.3f}   random pairs: {np.median(rcos):.3f}")
    print(f"  raw difference / row size: median {np.median(raw):.3f}  (p10 {np.percentile(raw, 10):.3f}, p90 {np.percentile(raw, 90):.3f})")
    print(f"  after one shared push:     median {np.median(left):.3f}  (p10 {np.percentile(left, 10):.3f}, p90 {np.percentile(left, 90):.3f})")
    print(f"  shared push size / typical row: {np.linalg.norm(push) / np.median(norm[b]):.3f}")


kind("leading space (hello -> _hello)", [(s, SP + s) for s in words if not s.startswith(SP) and SP + s in tok])
kind("capital first (hello -> Hello)", [(s, s[0].upper() + s[1:]) for s in words
                                        if s.lstrip(SP)[:1].islower() and (s[0].upper() + s[1:]) in tok and not s.startswith(SP)])
kind("capital first, with space (_hello -> _Hello)", [(s, SP + s[1].upper() + s[2:]) for s in words
                                        if s.startswith(SP) and s[1:2].islower() and (SP + s[1].upper() + s[2:]) in tok])
