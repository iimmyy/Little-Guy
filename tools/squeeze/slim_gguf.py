"""Copy only the tensors whose names match a regex into a small GGUF (architecture key only), e.g. to send just the
matched block matrices of a donor to another machine. Readers that look tensors up by name (squeeze2.py donors)
treat it like the full file for those tensors.

usage: slim_gguf.py <in.gguf> <out.gguf> <regex>
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py"))
import gguf

src, out, pat = sys.argv[1:4]
r = gguf.GGUFReader(src)
keep = [t for t in r.tensors if re.search(pat, t.name)]
w = gguf.GGUFWriter(out, r.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=r.endianess)
for t in keep:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in keep:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
print(f"{out}: {len(keep)} of {len(r.tensors)} tensors, {sum(int(t.n_bytes) for t in keep) / 1e9:.3f} GB")
