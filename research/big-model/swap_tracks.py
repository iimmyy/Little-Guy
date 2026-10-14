"""Build a variant of a GGUF brain with some tensors taken from another GGUF of the same model (same names and
shapes), e.g. our 2-bit 31B with attn_q back at 4-bit from Google's file. Metadata comes from the first brain.

usage: swap_tracks.py <gguf-py dir> <brain.gguf> <donor.gguf> <out.gguf> <regex> [<regex> ...]
"""
import re, sys
sys.path.insert(0, sys.argv[1])
import gguf

brain, donor, out = sys.argv[2:5]
pats = [re.compile(p) for p in sys.argv[5:]]
rb, rd = gguf.GGUFReader(brain), gguf.GGUFReader(donor)
dt = {t.name: t for t in rd.tensors}
swap = {t.name for t in rb.tensors if any(p.search(t.name) for p in pats)}
missing = [n for n in swap if n not in dt or tuple(dt[n].shape) != tuple(next(t for t in rb.tensors if t.name == n).shape)]
assert not missing, f"donor lacks or reshapes: {missing[:5]}"

w = gguf.GGUFWriter(out, rb.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=rb.endianess)
for f in rb.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
tensors = [dt[t.name] if t.name in swap else t for t in rb.tensors]
for t in tensors:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in tensors:
    w.write_tensor_data(t.data, tensor_endianess=rb.endianess)
w.close()
print(f"swapped {len(swap)} tensors from {donor}")
