"""A copy of one GGUF with chosen tensors taken from another: everything else (metadata, word tables, norms) stays
byte-identical. Used for clean comparisons, e.g. Google's own 4-bit file with only its block matrices replaced.

usage: swap_blocks.py <base.gguf> <donor.gguf> <out.gguf> <regex of tensor names to take from the donor> [--any-type]
The donor's tensor must have the same shape and type as the base's (same format, same bytes); with --any-type only
the same shape (e.g. a word table re-stored from Q6_K as Q4_0).
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py"))
import gguf

base_p, don_p, out_p, pat = sys.argv[1:5]
any_type = "--any-type" in sys.argv[5:]
base = gguf.GGUFReader(base_p)
don = {t.name: t for t in gguf.GGUFReader(don_p).tensors}
take = re.compile(pat)
w = gguf.GGUFWriter(out_p, base.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=base.endianess)
for f in base.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
ts, n = [], 0
for t in base.tensors:
    if take.search(t.name):
        d = don[t.name]
        if any_type:
            assert [int(x) for x in d.shape] == [int(x) for x in t.shape], f"{t.name}: different shape"
        else:
            assert d.tensor_type == t.tensor_type and d.data.nbytes == t.data.nbytes, f"{t.name}: different format"
        ts.append(d); n += 1
    else:
        ts.append(t)
for t in ts:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in ts:
    w.write_tensor_data(t.data, tensor_endianess=base.endianess)
w.close()
print(f"{out_p}: {n} tensors from the donor, {len(ts) - n} unchanged from the base")
