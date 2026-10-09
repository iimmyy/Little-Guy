"""C: skim, then zoom (SVD-softmax, Shim et al. 2017).
Skim: score every word with a low-resolution copy of the dictionary (rank-R SVD).
Zoom: rescore the top-N skim candidates with the full dictionary.
Question: how often is his true best word (and his top-5) inside the zoomed candidates?
"""
import numpy as np

from pathlib import Path
RUN = str(Path(__file__).resolve().parents[2] / ".run")
PERSONA_FILE = Path(__file__).resolve().parents[2] / "persona/little_guy.private.txt"
DATA = f"{RUN}/results/vocab-trim"
E = np.load(f"{DATA}/E.npy")                                   # 262144 x 2560
H = dict(np.load(f"{DATA}/H.npz"))

# main "directions" of the dictionary: eigenvectors of E^T E (2560 x 2560)
w, V = np.linalg.eigh((E.T @ E).astype(np.float64))
V = V[:, ::-1].astype(np.float32)                              # strongest directions first

print(f"{'rank':>4s} {'skim size':>9s} {'zoom':>5s} | " + "  ".join(f"{l[:3]:>6s}" for l in H) + " | top-5 kept (all)")
for R in (128, 256, 384):
    low = E @ V[:, :R]                                         # the low-resolution dictionary, 262144 x R
    skim_mb = low.size * 1.0 / 1e6                            # at 8-bit
    for N in (512, 1024, 2048):
        hits, top5 = [], []
        for lang, h in H.items():
            full = h @ E.T
            best = full.argmax(1)
            true5 = np.argsort(-full, axis=1)[:, :5]
            approx = (h @ V[:, :R]) @ low.T
            cand = np.argpartition(-approx, N, axis=1)[:, :N]
            hits.append(np.mean([b in c for b, c in zip(best, cand)]))
            top5.append(np.mean([np.isin(t, c).all() for t, c in zip(true5, cand)]))
        print(f"{R:4d} {skim_mb:7.0f}MB {N:5d} | " + "  ".join(f"{x:6.1%}" for x in hits) + f" | {np.mean(top5):6.1%}")
