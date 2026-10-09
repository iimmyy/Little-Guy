"""Fairer grade: what share of his word-choice probability does each shortcut keep? (temperature 1, Gemma softcap 30)"""
import sys
import numpy as np

from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
PERSONA_FILE = Path(__file__).resolve().parents[2] / "persona/little_guy.private.txt"
sys.path.insert(0, f"{RUN}/lcpp-src/gguf-py")
from gguf import GGUFReader
E = np.load(f"{RUN}/results/vocab-trim/E.npy"); H = dict(np.load(f"{RUN}/results/vocab-trim/H.npz")); CAP = 30.0

order = next(x for x in GGUFReader(f"{RUN}/weights/drafter/gemma-4-E4B-assistant-masked-q8_0.gguf").tensors if "ordering" in x.name).data.astype(np.int64)
sections = order.reshape(2048, 128); heads = E[sections].mean(1)
w, V = np.linalg.eigh((E.T @ E).astype(np.float64)); V = V[:, ::-1].astype(np.float32)
low = E @ V[:, :256]


def kept(h, cand_fn):
    out = []
    for x in h:
        z = CAP * np.tanh((E @ x) / CAP); p = np.exp(z - z.max()); p /= p.sum()
        out.append(p[cand_fn(x)].sum())
    return np.array(out)


setups = {
    "guesser sections, top-32 (3% read)": lambda x: sections[np.argsort(-(heads @ x))[:32]].ravel(),
    "guesser sections, top-64 (6% read)": lambda x: sections[np.argsort(-(heads @ x))[:64]].ravel(),
    "skim 256 + zoom 2048 (~12% read)":    lambda x: np.argpartition(-(low @ (x @ V[:, :256])), 2048)[:2048],
}
print(f"{'shortcut':36s} | " + "  ".join(f"{l[:3]:>6s}" for l in H) + " | worst word")
for name, fn in setups.items():
    ks = {l: kept(h, fn) for l, h in H.items()}
    print(f"{name:36s} | " + "  ".join(f"{k.mean():6.1%}" for k in ks.values()) + f" | {min(k.min() for k in ks.values()):6.1%}")
