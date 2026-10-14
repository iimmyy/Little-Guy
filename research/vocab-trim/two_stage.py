"""A cheaper dictionary preview in two stages, measured before building.

Today: score all 262,144 words against a 2-bit copy of the dictionary (Q2_K, 220 MB on brain C), keep 1,024,
score those at full quality. Here stage 1 scores all words against a low-dimensional copy (the dictionary's top-r
principal directions, 8-bit per value), keeps M, then stage 2 runs today's 2-bit preview on just those M and keeps
1,024. Reads r MB/1000 x 262 + M rows of the 2-bit copy instead of 220 MB.

Per setting: how often his whole dice table (temperature 0.8, top-k 64, top-p 0.95, min-p 0.05) ends up inside the
final 1,024, and how often the final 1,024 equal today's.

usage: two_stage.py DATA_NAME PREVIEW_GGUF [--rank 128,256,384] [--keep 2048,4096,8192]
"""
import sys
import numpy as np
from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
sys.path.insert(0, f"{RUN}/lcpp-src/gguf-py")
from gguf import GGUFReader
from gguf.quants import dequantize

SOFTCAP, TEMP, BATCH, SHORT = 30.0, 0.8, 256, 1024
args = sys.argv[1:]
DATA = f"{RUN}/results/vocab-trim/{args.pop(0)}"
GG = args.pop(0)
def opt(name, default):
    if name in args:
        i = args.index(name); v = tuple(int(x) for x in args[i + 1].split(",")); del args[i:i + 2]; return v
    return default
RANKS, KEEPS = opt("--rank", (128, 256, 384)), opt("--keep", (2048, 4096, 8192))

E = np.load(f"{DATA}/E.npy")
H = np.concatenate(list(dict(np.load(f"{DATA}/H.npz")).values()))
n = len(H)
ts = {t.name: t for t in GGUFReader(GG).tensors}
t = ts.get("token_embd_preview.weight") or ts["token_embd.weight"]
Q = dequantize(t.data, t.tensor_type).reshape(-1, 2560).astype(np.float32)
print(f"{n} thoughts; 2-bit preview {t.tensor_type.name} {t.n_bytes / 1e6:.0f} MB", flush=True)

w, V = np.linalg.eigh(E.T @ E)
V = V[:, ::-1].astype(np.float32)
low = {}
for r in RANKS:
    T = E @ V[:, :r]
    s = np.abs(T).max(1, keepdims=True) / 127.0
    low[r] = (np.round(T / s) * s).astype(np.float32)          # 8-bit per value, one scale per word
    print(f"rank {r}: stage-1 table {len(E) * r / 1e6:.0f} MB at 8 bits", flush=True)

res = {(r, m): [0, 0] for r in RANKS for m in KEEPS}
base_full = 0
for i in range(0, n, BATCH):
    Hb = H[i:i + BATCH]
    full_scores = Hb @ E.T
    prev = Hb @ Q.T
    S1 = {r: (Hb @ V[:, :r]) @ low[r].T for r in RANKS}
    for k in range(len(Hb)):
        lg = SOFTCAP * np.tanh(full_scores[k] / SOFTCAP)
        top = np.argpartition(-lg, 64)[:64]; top = top[np.argsort(-lg[top])]
        p = np.exp(lg[top] - lg[top[0]]); p /= p.sum()
        keep = max(min(np.searchsorted(np.cumsum(p), 0.95) + 1, int((p >= 0.05 * p[0]).sum())), 1)
        dice = top[:keep]
        today = np.argpartition(-prev[k], SHORT)[:SHORT]
        base_full += np.isin(dice, today).all()
        today_set = set(today.tolist())
        for r in RANKS:
            s1 = S1[r][k]
            for m in KEEPS:
                cand = np.argpartition(-s1, m)[:m]
                final = cand[np.argpartition(-prev[k][cand], SHORT)[:SHORT]]
                res[(r, m)][0] += np.isin(dice, final).all()
                res[(r, m)][1] += set(final.tolist()) == today_set
    if (i // BATCH) % 10 == 0:
        print(f"  {i + len(Hb)}/{n}", flush=True)

row2 = t.n_bytes / len(Q)
print(f"\ntoday (2-bit preview on all words): dice table inside {base_full}/{n}, reads {t.n_bytes / 1e6:.0f} MB")
for (r, m), (full, same) in res.items():
    mb = len(E) * r / 1e6 + m * row2 / 1e6
    print(f"rank {r:3d}, keep {m:5d}: dice table inside {full}/{n}, same 1,024 as today {same}/{n}, reads {mb:5.0f} MB")
