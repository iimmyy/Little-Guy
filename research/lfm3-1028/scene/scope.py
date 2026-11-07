"""Draw his inner picture: for a few problems, each point where the drawing puts it (grey) vs where the probe reads it
in his notepad at the cue word (colour), held-out folds only. usage: scope.py capture.npz out.png id1,id2,id3 [layer=7]"""
import os
import sys, numpy as np
sys.path.insert(0, os.environ["LITTLEGUY_ROOT"] + "/tools/mind/census")
from scene_probe import ridge_cv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
z = np.load(sys.argv[1]); out, want = sys.argv[2], sys.argv[3].split(",")
L = int(dict(a.split("=") for a in sys.argv[4:]).get("layer", 7))
s = z["kind"] == "cue"
X, Y, G, LET = z["X"][s, L].astype(np.float32), z["xy"][s], z["prob"][s], z["letter"][s]
P = ridge_cv(X, Y, G)
ids = list(z["ids"])
fig, axs = plt.subplots(1, len(want), figsize=(4.2 * len(want), 4.4))
for ax, w in zip(np.atleast_1d(axs), want):
    g = ids.index(w); m = G == g
    for (tx, ty), (px, py), l in zip(Y[m], P[m], LET[m]):
        ax.plot([tx, px], [ty, py], color="#bbb", lw=1, zorder=1)
        ax.scatter(tx, ty, s=140, color="#ddd", edgecolor="#888", zorder=2); ax.text(tx, ty, l, ha="center", va="center", fontsize=9)
        ax.scatter(px, py, s=140, color="#4a7bd0", zorder=3); ax.text(px, py, l, ha="center", va="center", color="white", fontsize=9)
    ax.set_title(w.replace("fresh/", ""), fontsize=10); ax.set_aspect("equal"); ax.grid(alpha=.3)
fig.suptitle(f"grey = where the drawing puts each point, blue = where his notepad has it (layer index {L}, held-out read)", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=130); print("saved", out)
