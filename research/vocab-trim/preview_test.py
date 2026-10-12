"""Lo-fi preview of the dictionary: score every word against a small, rough copy first, then score only
the shortlist at full quality. Lossless whenever his whole dice table makes the shortlist.

Rough copies (all start by rotating the dictionary onto its own main directions, found with SVD):
  rank-r int8   keep only the r strongest directions, 8 bits each        -> r bytes per word
  1-bit         keep all 2560 directions, only the sign (+ one scale)    -> 320 bytes per word
  2-bit         keep all 2560 directions, 4 levels (+ one step size)     -> 640 bytes per word
Full-quality rows cost 2560 x 4.5 bits = 1440 bytes each (Q4_0), so a shortlist of 1,024 is ~1.5 MB.

Dice table = the words his sampler can actually roll (top-k 64, then top-p 0.95, then llama.cpp's default
min-p 0.05, on softcapped scores). If the shortlist holds all of them, his dice roll is unchanged.

usage: preview_test.py [DATA_NAME] [--rank] [SHORTLIST ...]
  DATA_NAME  a folder in .run/results/vocab-trim made by read_collect.py (default: the Oct 8 data)
  --rank     also try the rank-r copies (they lost badly on the Oct 8 data)
"""
import sys, time
import numpy as np

from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
DATA = f"{RUN}/results/vocab-trim"
FULL_ROW = 2560 * 4.5 / 8                                    # bytes per dictionary row at Q4_0
FULL_MB = 378                                                # what the full scan reads today
SOFTCAP = 30.0                                               # gemma4.final_logit_softcapping
BATCH = 512                                                  # thoughts scored at a time, to fit in RAM

args = sys.argv[1:]
if args and not args[0].startswith("-") and not args[0].isdigit():
    DATA = f"{DATA}/{args.pop(0)}"
RANK = "--rank" in args
SHORT = tuple(int(x) for x in args if x.isdigit()) or (64, 256, 1024, 4096)

E = np.load(f"{DATA}/E.npy")                                 # 262144 x 2560
Hall = np.concatenate(list(dict(np.load(f"{DATA}/H.npz")).values()))
n_word = len(Hall)
print(f"{n_word} thoughts from {Path(DATA).name}", flush=True)


def dice_table(row):
    lg = SOFTCAP * np.tanh(row / SOFTCAP)
    top = np.argpartition(-lg, 64)[:64]
    top = top[np.argsort(-lg[top])]
    p = np.exp(lg[top] - lg[top[0]]); p /= p.sum()
    keep = np.searchsorted(np.cumsum(p), 0.95) + 1
    keep = min(keep, int((p >= 0.05 * p[0]).sum()))
    return top[:max(keep, 1)]


# the answer key, one batch at a time
t0 = time.time()
best, dice = np.empty(n_word, np.int64), []
for i in range(0, n_word, BATCH):
    s = Hall[i:i + BATCH] @ E.T
    best[i:i + BATCH] = s.argmax(1)
    dice += [dice_table(r) for r in s]
sizes = [len(d) for d in dice]
print(f"answer key in {time.time() - t0:.0f} s; dice table median {int(np.median(sizes))} words, largest {max(sizes)}", flush=True)

# rotate onto the dictionary's main directions (strongest first); scores don't change: (hV)·(eV) = h·e
w, V = np.linalg.eigh(E.T @ E)
V = V[:, ::-1].astype(np.float32)
R = E @ V
del E
Hr = Hall @ V

copies = []
if RANK:
    for r in (128, 256, 512):
        s = np.abs(R[:, :r]).max(1, keepdims=True) / 127
        copies.append((f"rank-{r} int8", len(R) * r / 1e6, r, (np.round(R[:, :r] / s) * s).astype(np.float32)))
copies.append(("1-bit", len(R) * 2560 / 8 / 1e6, None, (np.sign(R) * np.abs(R).mean(1, keepdims=True)).astype(np.float32)))
step = R.std(1, keepdims=True)
copies.append(("2-bit", len(R) * 2560 / 4 / 1e6, None, ((np.clip(np.floor(R / step), -2, 1) + 0.5) * step).astype(np.float32)))
del R

print(f"\n{'rough copy':14s} {'size':>7s} | " + " | ".join(f"shortlist {k:<6d} best   dice   MB" for k in SHORT))
for name, mb, r, Q in copies:
    hit1, hitd = np.zeros(len(SHORT)), np.zeros(len(SHORT))
    for i in range(0, n_word, BATCH):
        h = Hr[i:i + BATCH, :r] if r else Hr[i:i + BATCH]
        order = np.argsort(-(h @ Q.T), axis=1)[:, :max(SHORT)]
        for j, k in enumerate(SHORT):
            for n, sl in enumerate(order[:, :k]):
                hit1[j] += best[i + n] in sl
                hitd[j] += np.isin(dice[i + n], sl).all()
    cells = [f"{hit1[j] / n_word:7.3%} {hitd[j] / n_word:7.3%} {mb + k * FULL_ROW / 1e6:4.0f}" for j, k in enumerate(SHORT)]
    print(f"{name:14s} {mb:4.0f} MB | " + " | ".join(cells), flush=True)
    print(f"{'':14s} {'misses':>7s} | " + " | ".join(f"{'':16s} {n_word - int(hitd[j]):5d} dice" for j in range(len(SHORT))), flush=True)
print(f"\ntoday's full scan reads {FULL_MB} MB per word; 1 miss in {n_word} words = {1 / n_word:.3%}")
