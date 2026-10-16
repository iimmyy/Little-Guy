"""Bytes a decoder reads per generated word: every weight except the per-layer token table (one row is looked up
per word) and, for E4B-style models, the token table counted once as the tied output head (full read per word,
or its preview when the file has one). usage: bytes_per_word.py <gguf> [...]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py"))
import gguf
for p in sys.argv[1:]:
    ts = {t.name: int(t.n_bytes) for t in gguf.GGUFReader(p).tensors}
    head = ts.get("token_embd_preview.weight") or ts.get("output.weight") or ts.get("token_embd.weight", 0)
    body = sum(b for n, b in ts.items() if n not in ("per_layer_token_embd.weight", "token_embd.weight",
                                                     "token_embd_preview.weight", "output.weight"))
    print(f"{Path(p).name:32} file {sum(ts.values()) / 1e9:5.2f} GB   per word {(body + head) / 1e9:5.2f} GB")
