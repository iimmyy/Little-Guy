#!/usr/bin/env python3
"""Find where Gemma E4B keeps the copying skill: "this came before, so what followed it last time comes next".

Gemma reads random word sequences that repeat once ([bos] R R). On the second copy, a copying head (an induction head)
looks back at the word that followed the current word the first time; a previous-word head looks at the word just
before. Every head in all 42 rooms gets both scores. Then the proof: silence the top copying heads and Gemma should
lose the skill (its loss on the second copy jumps), while silencing as many random heads should not.

usage: find_skill.py <model.safetensors> <out.json> [n_seq] [len]
"""
import json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma_stream import Gemma, N_LAYER, HEADS


def copy_losses(logits, ids, L):
    """mean next-word loss on the first copy (cannot be guessed) and on the second copy (can be copied)"""
    lp = F.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten(), reduction="none").view(ids.shape[0], -1)
    return float(lp[:, 1:L].mean()), float(lp[:, L + 1:2 * L].mean())


def main():
    path, out = sys.argv[1], sys.argv[2]
    n, L = (int(sys.argv[3]) if len(sys.argv) > 3 else 16), (int(sys.argv[4]) if len(sys.argv) > 4 else 50)
    rng = np.random.default_rng(0)
    R = torch.tensor(rng.integers(1000, 20000, size=(n, L)))                 # ordinary vocabulary, no special tokens
    ids = torch.cat([torch.full((n, 1), 2), R, R], 1)                       # [bos] R R; R starts at 1, the copy at L+1
    g = Gemma(path)
    ind, prev = np.zeros((N_LAYER, HEADS)), np.zeros((N_LAYER, HEADS))
    q = torch.arange(L + 2, 2 * L + 1)                                      # second-copy positions that have a "next"
    target = q - L + 1                                                      # where the word after this one sat last time

    def hook(il, att):
        ind[il] = att[:, :, q, target].mean((0, 2)).numpy()
        prev[il] = att[:, :, q, q - 1].mean((0, 2)).numpy()

    t0 = time.time()
    with torch.no_grad():
        base = copy_losses(g.run(ids, hook=hook), ids, L)
    print(f"Gemma, nothing silenced: loss first copy {base[0]:.2f}, second copy {base[1]:.3f}  ({time.time() - t0:.0f}s)",
          flush=True)
    order = np.dstack(np.unravel_index(np.argsort(-ind, axis=None), ind.shape))[0]
    print("top copying heads (room, head, score; previous-word score of the same head):")
    for il, hh in order[:12]:
        print(f"  room {il:2d} head {hh}  copy {ind[il, hh]:.3f}  previous-word {prev[il, hh]:.3f}")
    porder = np.dstack(np.unravel_index(np.argsort(-prev, axis=None), prev.shape))[0]
    print("top previous-word heads:", [(int(a), int(b), round(float(prev[a, b]), 3)) for a, b in porder[:8]], flush=True)

    tests = {}
    for k in (4, 8):
        top = {}
        for il, hh in order[:k]:
            top.setdefault(int(il), []).append(int(hh))
        with torch.no_grad():
            tests[f"top{k}"] = dict(heads=top, loss=copy_losses(g.run(ids, ablate=top), ids, L))
        for r in range(2):
            pick = rng.choice(N_LAYER * HEADS, k, replace=False)
            rand = {}
            for p in pick:
                rand.setdefault(int(p // HEADS), []).append(int(p % HEADS))
            with torch.no_grad():
                tests[f"random{k}_{r}"] = dict(heads=rand, loss=copy_losses(g.run(ids, ablate=rand), ids, L))
        for name in [f"top{k}", f"random{k}_0", f"random{k}_1"]:
            print(f"silence {name:10s}: first copy {tests[name]['loss'][0]:.2f}, second copy {tests[name]['loss'][1]:.3f}",
                  flush=True)
    json.dump(dict(n=n, L=L, base=base, induction=ind.tolist(), previous=prev.tolist(), tests=tests),
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
