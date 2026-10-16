"""Which matching settings generalize? For a sample of matrices (every group, several depths), compute the input
statistics once and match each donor type under each setting; report output error on the FIT documents and on the
HELD-OUT documents (whole conversations and articles the matching never saw), stock rounding vs matched.

usage: match_settings.py <ref bf16> <act dir> <out.json> <type=donor.gguf> [type=donor.gguf ...]
env: LG_SET_LAYERS (default 2,13,24,35), LG_SET_DAMPS (default 0.01,0.1), LG_SET_ORDERS (default plain,act)
"""
import json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import match as M
from gguf.quants import dequantize

ref_p, act, out_p = sys.argv[1], sys.argv[2], Path(sys.argv[3])
donors = dict(a.split("=", 1) for a in sys.argv[4:])
layers = [int(x) for x in os.environ.get("LG_SET_LAYERS", "2,13,24,35").split(",")]
damps = [float(x) for x in os.environ.get("LG_SET_DAMPS", "0.01,0.1").split(",")]
orders = os.environ.get("LG_SET_ORDERS", "plain,act").split(",")
ref = M.load(ref_p)
don = {ty: M.load(p) for ty, p in donors.items()}
res = json.loads(out_p.read_text()) if out_p.exists() else {}
for L in layers:
    for g in M.ACT:
        name = f"blk.{L}.{g}.weight"
        if name not in ref or (name in res and len(res[name]) == len(donors) * len(damps) * len(orders)):
            continue
        f = M.hessian(act, name)
        n_col = int(ref[name].shape[0])
        t0 = time.time()
        G = M.grams(f, n_col)
        W = dequantize(ref[name].data, ref[name].tensor_type).reshape(-1, n_col).astype(np.float32)
        res.setdefault(name, {})
        for ty, d in don.items():
            dt = d[name]
            raw = np.asarray(dt.data).reshape(W.shape[0], -1)
            a, b, c0, qmax = M.grid(raw, dt.tensor_type, n_col)
            E0 = a * c0 + b - W
            for dp in damps:
                for o in orders:
                    c1 = M.match(W, G["fit"], a, b, c0, qmax, dp, order=None if o == "plain" else o)
                    E1 = a * c1 + b - W
                    res[name][f"{ty}|damp{dp}|{o}"] = {
                        "fit_stock": M.out_err(E0, G["fit"]), "fit_matched": M.out_err(E1, G["fit"]),
                        "hold_stock": M.out_err(E0, G["hold"]), "hold_matched": M.out_err(E1, G["hold"])}
        out_p.write_text(json.dumps(res, indent=1))
        M.log(f"{name} ({n_col} inputs, rows {G['n_fit']}+{G['n_hold']}): {time.time() - t0:.0f} s")
