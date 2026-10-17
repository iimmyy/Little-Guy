"""Fingerprint a GGUF file so two copies can be compared without moving them: sha256 of the whole file, of every
tensor's bytes, and of every metadata value. Writes JSON next to the given path (or to --out).

usage: PYTHONPATH=<engine>/gguf-py python gguf_fingerprint.py model.gguf [--out x.json]
       python gguf_fingerprint.py --compare a.json b.json
"""
import hashlib, json, sys
from pathlib import Path


def fingerprint(path):
    from gguf import GGUFReader
    whole = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            whole.update(b)
    r = GGUFReader(path)
    meta = {}
    for k, fld in r.fields.items():
        h = hashlib.sha256()
        for i in fld.data:
            h.update(fld.parts[i].tobytes())
        meta[k] = h.hexdigest()
    tensors = {t.name: {"type": t.tensor_type.name, "shape": [int(x) for x in t.shape],
                        "sha256": hashlib.sha256(t.data.tobytes()).hexdigest()} for t in r.tensors}
    return {"file": Path(path).name, "bytes": Path(path).stat().st_size, "sha256": whole.hexdigest(),
            "meta": meta, "tensors": tensors}


def compare(a, b):
    A, B = (json.load(open(x)) for x in (a, b))
    print(f"whole file: {'IDENTICAL' if A['sha256'] == B['sha256'] else 'different'} "
          f"({A['bytes']:,} vs {B['bytes']:,} B)")
    for part in ("meta", "tensors"):
        ka, kb = set(A[part]), set(B[part])
        same = [k for k in ka & kb if A[part][k] == B[part][k]]
        diff = sorted(k for k in ka & kb if A[part][k] != B[part][k])
        print(f"{part}: {len(same)}/{len(ka | kb)} identical; differ {diff[:10]}; "
              f"only in first {sorted(ka - kb)[:5]}; only in second {sorted(kb - ka)[:5]}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else sys.argv[1] + ".fingerprint.json"
        json.dump(fingerprint(sys.argv[1]), open(out, "w"), indent=1)
        print("wrote", out)
