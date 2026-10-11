"""Distortion meter: how far do 2-bit (Q2_K) weights land from the master, default compressor vs careful?

Compresses real tensors from his QAT master with llama.cpp's own quantize_q2_K, then expands them back and
compares with the original. No text is involved; this measures only the weights.
  relative error = sum((compressed - original)^2) / sum(original^2), in dB (lower is closer)
LG_Q2K_CAREFUL is read once per process, so each setting runs in its own process.

Usage: python distortion.py            (prints the table; runs both compressors)
"""
import ctypes, os, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / ".run"
sys.path.insert(0, str(RUN / "lcpp-src/gguf-py"))
from gguf import GGUFReader

LIB = RUN / "lcpp-src/build-mac/bin/libggml-base.dylib"
MASTER = RUN / "weights/masters/qat-bf16.gguf"
NAMES = [f"blk.{b}.{t}.weight" for b in (0, 20, 41) for t in ("ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_output")]
NAMES += ["token_embd.weight"]
ROWS = 512                                   # rows sampled per tensor (spread evenly), enough for a stable number


def load(r, name):
    t = next(t for t in r.tensors if t.name == name)
    raw = np.asarray(t.data)
    if raw.dtype == np.uint16 or t.tensor_type.name == "BF16":       # bf16 bits -> float32
        a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        a = raw.astype(np.float32)
    a = a.reshape(-1, a.shape[-1])
    pick = np.linspace(0, a.shape[0] - 1, min(ROWS, a.shape[0])).astype(int)
    return np.ascontiguousarray(a[pick])


def errors():
    lib = ctypes.CDLL(str(LIB))
    lib.quantize_q2_K.restype = ctypes.c_size_t
    lib.quantize_q2_K.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.dequantize_row_q2_K.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    r = GGUFReader(MASTER)
    out = {}
    for name in NAMES:
        x = load(r, name)
        nrow, n = x.shape
        q = np.zeros(nrow * n // 256 * 84, np.uint8)
        y = np.zeros_like(x)
        lib.quantize_q2_K(x.ctypes.data, q.ctypes.data, nrow, n, None)
        lib.dequantize_row_q2_K(q.ctypes.data, y.ctypes.data, nrow * n)
        out[name] = float(((y - x) ** 2).sum() / (x ** 2).sum())
    return out


CONFIGS = {  # label: environment for llama.cpp's quantize_q2_K
    "default": {"LG_Q2K_CAREFUL": "0"},
    "careful v1": {"LG_Q2K_CAREFUL": "1", "LG_Q2K_ITERS": "4"},
    "v2, 3 starts": {"LG_Q2K_CAREFUL": "2", "LG_Q2K_STARTS": "3", "LG_Q2K_RADIUS": "2"},
    "v2, 6 starts": {"LG_Q2K_CAREFUL": "2", "LG_Q2K_STARTS": "6", "LG_Q2K_RADIUS": "3"},
    "v2, 6 starts, full": {"LG_Q2K_CAREFUL": "2", "LG_Q2K_STARTS": "6", "LG_Q2K_RADIUS": "-1"},
}

if __name__ == "__main__":
    if os.environ.get("DISTORTION_CHILD"):
        print(repr(errors()))
        sys.exit()
    import time
    runs, secs = {}, {}
    for label, cfg in CONFIGS.items():
        t0 = time.time()
        env = dict(os.environ, DISTORTION_CHILD="1", **cfg)
        runs[label] = eval(subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True, check=True).stdout)
        secs[label] = time.time() - t0
        print(f"{label}: {secs[label]:.0f} s", flush=True)
    labels = list(CONFIGS)
    print(f"\n{'tensor':28s} " + " ".join(f"{l:>19s}" for l in labels))
    for name in NAMES:
        print(f"{name:28s} " + " ".join(f"{10 * np.log10(runs[l][name]):17.2f}dB" for l in labels))
    avg = {l: np.mean([10 * np.log10(v) for v in runs[l].values()]) for l in labels}
    print(f"{'average':28s} " + " ".join(f"{avg[l]:17.2f}dB" for l in labels))
    print(f"{'closer than default':28s} " + " ".join(f"{avg['default'] - avg[l]:17.2f}dB" for l in labels))
