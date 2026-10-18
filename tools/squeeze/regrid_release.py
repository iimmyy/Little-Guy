#!/usr/bin/env python3
"""Apply the E4B method to any Google QAT 4-bit release, given its 16-bit QAT original.

  1. grid check: for every weight tensor Google stores quantized or at 16 bits, the share of 32-value blocks in the
     16-bit original that hold <= 16 distinct values (a 4-bit grid) - measured, not assumed
  2. right scale: Google's 4-bit (Q4_0) tensors re-stored as Q4_0 on their own grid (grid_q4_0.py); same size
  3. grid: also the tensors Google keeps at 6 bits (Q6_K) or 16 bits, re-stored as grid Q4_0 when they pass the check

Writes <out>/<tag>-rightscale.gguf and <out>/<tag>-grid4.gguf (+ a .json listing what changed).
usage: regrid_release.py <google release.gguf> <qat bf16.gguf> <out dir> <tag> [--check-only]
"""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / ".run/lcpp-src/gguf-py"))
import numpy as np
import gguf

google, bf16, out, tag = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
check_only = "--check-only" in sys.argv
out.mkdir(parents=True, exist_ok=True)
T = gguf.GGMLQuantizationType
g = gguf.GGUFReader(google)
b = {t.name: t for t in gguf.GGUFReader(bf16).tensors}


def on_grid_share(t, max_blocks=200_000):
    """share of 32-blocks with <= 16 distinct values, on an even sample of up to max_blocks blocks"""
    a = np.asarray(t.data).reshape(-1, 32)              # a view of the file: sample blocks first, convert only those
    if len(a) > max_blocks:
        a = a[np.linspace(0, len(a) - 1, max_blocks).astype(int)]
    if t.tensor_type == T.BF16:
        a = (a.astype(np.uint32) << 16).view(np.float32)
    a = a.astype(np.float32)
    s = np.sort(a, axis=1)
    distinct = 1 + (np.diff(s, axis=1) != 0).sum(axis=1)
    return float((distinct <= 16).mean())


rows, q4, wide = [], [], []
for t in g.tensors:
    kind = t.tensor_type.name
    if kind == "F32" or t.name not in b or len(t.shape) < 2 or int(t.shape[0]) % 32:
        continue
    share = on_grid_share(b[t.name])
    rows.append({"name": t.name, "google": kind, "bytes": int(t.n_bytes), "on_grid": round(share, 4)})
    if kind == "Q4_0":
        q4.append(t.name)
    elif share >= 0.99:
        wide.append(t.name)
summary = {}
for r in rows:
    k = r["google"]; s = summary.setdefault(k, [0, 0, 1.0])
    s[0] += 1; s[1] += r["bytes"]; s[2] = min(s[2], r["on_grid"])
print(f"{tag}: Google's tensor formats (count, bytes, lowest on-grid share among them)")
for k, (n, by, lo) in summary.items():
    print(f"  {k:6} {n:4d} tensors {by / 1e9:7.3f} GB   on grid >= {lo:.4f}")
print("  kept at more than 4 bits by Google but on the grid:", [r for r in rows if r["name"] in wide][:8])
json.dump(rows, open(out / f"{tag}-gridcheck.json", "w"), indent=1)
if check_only:
    sys.exit(0)


def run(*a):
    print("$", " ".join(map(str, a)), flush=True)
    subprocess.run([sys.executable, *map(str, a)], check=True)


def build(name, names, base, extra=()):
    donor = out / f"{tag}-{name}-donor.gguf"
    run(HERE / "grid_q4_0.py", bf16, donor, *names)
    run(HERE / "swap_blocks.py", base, donor, out / f"{tag}-{name}.gguf", "^(" + "|".join(map(lambda n: n.replace(".", r"\."), names)) + ")$", *extra)
    donor.unlink()
    json.dump({"base": str(base), "regridded": names}, open(out / f"{tag}-{name}.json", "w"), indent=1)


build("rightscale", q4, google)                                   # same format and size as Google's
build("grid4", wide, out / f"{tag}-rightscale.gguf", ("--any-type",))  # + the wide tables at 4 bits on their grid
for f in (f"{tag}-rightscale.gguf", f"{tag}-grid4.gguf"):
    print(f, (out / f).stat().st_size, "bytes")
