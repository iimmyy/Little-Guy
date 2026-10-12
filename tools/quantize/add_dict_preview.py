"""Add a dictionary preview (token_embd_preview.weight) to a Gemma 4 GGUF.

The preview is a rough low-bit copy of the dictionary. With it, the brain skims all 262,144 words cheaply, keeps a
shortlist, and scores only those at full quality (see the fork's src/models/gemma4.cpp, LG_DICT_PREVIEW).
On brain C the 2-bit preview kept his whole dice table on every one of 12,733 words (research/vocab-trim).

The preview is copied from another GGUF's token_embd, made with llama-quantize, for example:
  llama-quantize --allow-requantize --tensor-type token_embd=q2_k --token-embedding-type q2_k brain.gguf rough.gguf Q2_K

Usage: python add_dict_preview.py <gguf-py dir> <brain.gguf> <rough.gguf> <out.gguf>
"""
import sys
sys.path.insert(0, sys.argv[1])
import gguf

brain, rough, out = sys.argv[2:5]
r = gguf.GGUFReader(brain)
preview = next(t for t in gguf.GGUFReader(rough).tensors if t.name == "token_embd.weight")
full = next(t for t in r.tensors if t.name == "token_embd.weight")
assert tuple(preview.shape) == tuple(full.shape), (preview.shape, full.shape)
assert not any(t.name == "token_embd_preview.weight" for t in r.tensors), "already has a preview"

w = gguf.GGUFWriter(out, r.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=r.endianess)
for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

tensors = list(r.tensors) + [preview]
names = [t.name for t in r.tensors] + ["token_embd_preview.weight"]
for name, t in zip(names, tensors):
    w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in tensors:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
print(f"{out}: added token_embd_preview.weight ({preview.tensor_type.name}, {preview.n_bytes / 1e6:.0f} MB)")
