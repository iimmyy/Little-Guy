"""A: our own dictionary sections (k-means on his main dictionary).
B: guaranteed-exact search: open sections in order of their best possible score, stop when no
   unopened section can beat the best word found so far.

Bound for section c (centre m, width r = farthest entry from m):  h·e <= h·m + |h|·r   (any entry e in c)
"""
import sys, time
import numpy as np

from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
PERSONA_FILE = Path(__file__).resolve().parents[2] / "persona/little_guy.private.txt"
DATA = f"{RUN}/results/vocab-trim"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
E = np.load(f"{DATA}/E.npy")                                  # 262144 x 2560 dictionary
H = dict(np.load(f"{DATA}/H.npz"))                            # hidden states per language
rng = np.random.default_rng(0)
e2 = (E * E).sum(1)


def assign(C):
    lab = np.empty(len(E), np.int64)
    c2 = (C * C).sum(1)
    for i in range(0, len(E), 32768):                        # nearest centre, in chunks to save RAM
        lab[i:i + 32768] = (c2[None, :] - 2 * E[i:i + 32768] @ C.T).argmin(1)
    return lab


# --- A. k-means
t0 = time.time()
C = E[rng.choice(len(E), K, replace=False)].copy()
for it in range(15):
    lab = assign(C)
    sums = np.zeros_like(C); np.add.at(sums, lab, E)
    cnt = np.bincount(lab, minlength=K)
    empty = cnt == 0
    C[~empty] = sums[~empty] / cnt[~empty, None]
    C[empty] = E[rng.choice(len(E), empty.sum(), replace=False)]   # re-seed empty sections
lab = assign(C)
cnt = np.bincount(lab, minlength=K)
width = np.zeros(K, np.float32)
dist = np.sqrt(np.maximum(e2 + (C * C).sum(1)[lab] - 2 * np.einsum("ij,ij->i", E, C[lab]), 0))
np.maximum.at(width, lab, dist)
print(f"k-means {K} sections in {time.time() - t0:.0f} s | section sizes min {cnt.min()} median {int(np.median(cnt))} max {cnt.max()}")
np.savez(f"{DATA}/sections_{K}.npz", centres=C, labels=lab, width=width)

# --- evaluate
print(f"{'language':9s} {'words':>5s} | top-32 by middle | top-32 by best-possible | exact: avg sections / avg share of dictionary read / worst")
for lang, h in H.items():
    scores = h @ E.T                                         # full dictionary (the answer key)
    best = scores.argmax(1)
    mid = h @ C.T
    bound = mid + np.linalg.norm(h, axis=1, keepdims=True) * width[None, :]
    secmax = np.full((len(h), K), -np.inf, np.float32)
    for k in range(K):                                       # best real score inside each section
        m = lab == k
        if m.any():
            secmax[:, k] = scores[:, m].max(1)
    hit_mid = np.mean([lab[b] in np.argsort(-s)[:32] for b, s in zip(best, mid)])
    hit_bnd = np.mean([lab[b] in np.argsort(-u)[:32] for b, u in zip(best, bound)])
    opened, read = [], []
    for i in range(len(h)):
        order = np.argsort(-bound[i])
        run = np.maximum.accumulate(secmax[i, order])
        # open section j while its best possible score could still beat the best found so far
        stop = np.argmax(bound[i, order][1:] <= run[:-1]) + 1 if (bound[i, order][1:] <= run[:-1]).any() else K
        opened.append(stop)
        read.append(cnt[order[:stop]].sum() / len(E))
    print(f"{lang:9s} {len(h):5d} | {hit_mid:15.2%} | {hit_bnd:22.2%} | {np.mean(opened):7.0f} / {np.mean(read):6.1%} / {max(read):6.1%}")
