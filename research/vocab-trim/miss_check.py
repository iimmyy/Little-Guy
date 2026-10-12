"""How much do the preview's dice-table misses matter? For the 2-bit rough copy at one shortlist size, find every
thought whose dice table didn't fully make the shortlist, and report the chance he'd actually have rolled
a missing word (after top-k 64, top-p 0.95, min-p 0.05 and temperature 0.8, like his chats).

usage: miss_check.py DATA_NAME [SHORTLIST]   (default 32768)
"""
import sys
import numpy as np

from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
DATA = f"{RUN}/results/vocab-trim/{sys.argv[1]}"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
SOFTCAP, TEMP, BATCH = 30.0, 0.8, 512

E = np.load(f"{DATA}/E.npy")
H = np.concatenate(list(dict(np.load(f"{DATA}/H.npz")).values()))
w, V = np.linalg.eigh(E.T @ E)
V = V[:, ::-1].astype(np.float32)
R = E @ V
step = R.std(1, keepdims=True)
Q = ((np.clip(np.floor(R / step), -2, 1) + 0.5) * step).astype(np.float32)
del R
Hr = H @ V

lost = []                                                    # chance of rolling a word the shortlist lost, per miss
for i in range(0, len(H), BATCH):
    full = H[i:i + BATCH] @ E.T
    rough = Hr[i:i + BATCH] @ Q.T
    for f, r in zip(full, rough):
        lg = SOFTCAP * np.tanh(f / SOFTCAP)
        top = np.argpartition(-lg, 64)[:64]
        top = top[np.argsort(-lg[top])]
        p = np.exp(lg[top] - lg[top[0]]); p /= p.sum()
        keep = min(np.searchsorted(np.cumsum(p), 0.95) + 1, int((p >= 0.05 * p[0]).sum()))
        dice = top[:max(keep, 1)]
        q = np.exp((lg[dice] - lg[dice[0]]) / TEMP); q /= q.sum()  # what he actually rolls with
        cut = -np.partition(-r, K - 1)[K - 1]                      # shortlist = rough score >= the K-th best
        out = r[dice] < cut
        if out.any():
            lost.append(q[out].sum())

lost = np.array(lost)
print(f"{len(H)} thoughts, shortlist {K}: {len(lost)} dice tables missed a word")
if len(lost):
    print(f"chance the roll hit a missing word, per miss: median {np.median(lost):.2%}, largest {lost.max():.2%}")
    print(f"over the whole text: {lost.sum():.2f} words expected to differ, 1 in {len(H) / max(lost.sum(), 1e-9):,.0f}")
