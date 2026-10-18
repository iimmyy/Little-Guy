#!/usr/bin/env python3
"""Compare two GGUF files tensor by tensor: format per tensor, and whether the stored bytes are identical.
usage: gguf_compare.py <a.gguf> <b.gguf> [name a] [name b]"""
import hashlib, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py"))
import gguf
a, b = gguf.GGUFReader(sys.argv[1]), gguf.GGUFReader(sys.argv[2])
na, nb = (sys.argv[3:5] + ["A", "B"])[:2] if len(sys.argv) > 3 else ("A", "B")
ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
h = lambda t: hashlib.sha256(memoryview(t.data).cast("B")).hexdigest()
same_type = Counter(); diff_type = Counter(); same_bytes = 0; same_bytes_size = 0; total = 0; rows = []
for n in sorted(set(ta) | set(tb)):
    if n not in ta or n not in tb:
        rows.append((n, ta.get(n) and ta[n].tensor_type.name, tb.get(n) and tb[n].tensor_type.name, "only in one")); continue
    x, y = ta[n], tb[n]; total += 1
    if x.tensor_type == y.tensor_type:
        same_type[x.tensor_type.name] += 1
        if h(x) == h(y): same_bytes += 1; same_bytes_size += x.n_bytes
        else: rows.append((n, x.tensor_type.name, y.tensor_type.name, "same format, different bytes"))
    else:
        diff_type[(x.tensor_type.name, y.tensor_type.name)] += 1
        rows.append((n, x.tensor_type.name, y.tensor_type.name, "different format"))
print(f"{na} vs {nb}: {total} tensors in both")
print(f"  same format: {dict(same_type)}")
print(f"  different format ({na} -> {nb}): {dict(diff_type)}")
print(f"  byte-identical tensors: {same_bytes} of {total} ({same_bytes_size / 1e9:.3f} GB)")
for r in rows[:25]: print("   ", *r)
if len(rows) > 25: print(f"    ... {len(rows) - 25} more")
