"""Which 2-bit format should the board's dictionary preview use? Same test as preview_test.py + miss_check.py,
but the rough copy is a real ggml format read from a GGUF's token_embd (no rotation, so the board can use
its existing kernels), next to the custom rotated 2-bit copy from the first test.

For each rough copy and shortlist: top word found, whole dice table found, and the number of words expected to
come out differently in his chats (temperature 0.8, top-k 64, top-p 0.95, min-p 0.05).

usage: preview_format.py DATA_NAME ROUGH_GGUF [ROUGH_GGUF ...] [--custom] [--short 16384,32768]
"""
import sys
import numpy as np

from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
sys.path.insert(0, f"{RUN}/lcpp-src/gguf-py")
from gguf import GGUFReader
from gguf.quants import dequantize

FULL_ROW = 551e6 / 262144                                    # bytes per full-quality row (brain C: Q6_K)
SOFTCAP, TEMP, BATCH = 30.0, 0.8, 512

args = sys.argv[1:]
DATA = f"{RUN}/results/vocab-trim/{args.pop(0)}"
SHORT = (16384, 32768)
if "--short" in args:
    i = args.index("--short"); SHORT = tuple(int(x) for x in args[i + 1].split(",")); del args[i:i + 2]
CUSTOM = "--custom" in args
ggufs = [a for a in args if a.endswith(".gguf")]

E = np.load(f"{DATA}/E.npy")
H = np.concatenate(list(dict(np.load(f"{DATA}/H.npz")).values()))
n = len(H)

# answer key: his dice table and what he rolls with, per thought
dice, roll = [], []
for i in range(0, n, BATCH):
    for f in H[i:i + BATCH] @ E.T:
        lg = SOFTCAP * np.tanh(f / SOFTCAP)
        top = np.argpartition(-lg, 64)[:64]
        top = top[np.argsort(-lg[top])]
        p = np.exp(lg[top] - lg[top[0]]); p /= p.sum()
        keep = max(min(np.searchsorted(np.cumsum(p), 0.95) + 1, int((p >= 0.05 * p[0]).sum())), 1)
        q = np.exp((lg[top[:keep]] - lg[top[0]]) / TEMP)
        dice.append(top[:keep]); roll.append(q / q.sum())
print(f"{n} thoughts from {Path(DATA).name}", flush=True)

copies = []
for g in ggufs:
    t = next(t for t in GGUFReader(g).tensors if t.name == "token_embd.weight")
    Q = dequantize(t.data, t.tensor_type).reshape(-1, 2560).astype(np.float32)
    copies.append((f"{Path(g).stem} ({t.tensor_type.name})", t.n_bytes / 1e6, None, Q))
if CUSTOM:
    w, V = np.linalg.eigh(E.T @ E)
    V = V[:, ::-1].astype(np.float32)
    R = E @ V
    step = R.std(1, keepdims=True)
    copies.append(("custom 2-bit, rotated", len(E) * 2560 / 4 / 1e6, V, ((np.clip(np.floor(R / step), -2, 1) + 0.5) * step).astype(np.float32)))
    del R
del E

for name, mb, V, Q in copies:
    Hq = H @ V if V is not None else H
    top1 = np.zeros(len(SHORT)); full = np.zeros(len(SHORT)); differ = np.zeros(len(SHORT))
    for i in range(0, n, BATCH):
        for k, r in enumerate(Hq[i:i + BATCH] @ Q.T):
            d, q = dice[i + k], roll[i + k]
            for j, K in enumerate(SHORT):
                inside = r[d] >= -np.partition(-r, K - 1)[K - 1]
                top1[j] += inside[0]; full[j] += inside.all(); differ[j] += q[~inside].sum()
    print(f"\n{name}: {mb:.0f} MB", flush=True)
    for j, K in enumerate(SHORT):
        print(f"  shortlist {K:6d}: top word {top1[j] / n:8.3%}  dice table {full[j] / n:8.3%}  "
              f"differ 1 in {n / max(differ[j], 1e-9):>9,.0f}  reads {mb + K * FULL_ROW / 1e6:4.0f} MB (today 551)", flush=True)
