"""Oven check: the vector q8_K 4x4 packing must give the same bytes as llama.cpp's generic one, and be faster.

Packs 4 rows of activations into the q8_Kx4 layout that the 2-bit trays read, both ways, and compares every byte
(scales, codes and 16-value sums). Includes the edge cases: all zeros, a tie between +x and -x, one huge negative value.

Usage: python pack_check.py [path to libggml-cpu]      (default: the Mac build)
"""
import ctypes, sys, time
from pathlib import Path

import numpy as np

LIB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / ".run/lcpp-src/build-mac/bin/libggml-cpu.dylib"
lib = ctypes.CDLL(str(LIB))
fast, ref = lib.ggml_quantize_mat_q8_K_4x4, lib.ggml_quantize_mat_q8_K_4x4_generic
for f in (fast, ref):
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
BLOCK = 4 * 4 + 256 * 4 + 64 * 2            # block_q8_Kx4: 4 floats, 1024 codes, 64 sums


def pack(f, x):
    k = x.shape[1]
    out = np.zeros(k // 256 * BLOCK, np.uint8)
    f(x.ctypes.data, out.ctypes.data, k)
    return out


rng = np.random.default_rng(0)
cases = {f"random x{s}": (rng.standard_normal((4, 2560)) * s).astype(np.float32) for s in (1e-3, 1, 50)}
cases["heavy tails"] = (rng.standard_t(2, (4, 10240))).astype(np.float32)
z = np.zeros((4, 512), np.float32); cases["all zeros"] = z
t = rng.standard_normal((4, 256)).astype(np.float32); t[:, 7] = 5; t[:, 3] = -5; cases["tie, -x first"] = t
t2 = t.copy(); t2[:, 3] = 5; t2[:, 7] = -5; cases["tie, +x first"] = t2
h = rng.standard_normal((4, 256)).astype(np.float32); h[2, 100] = -1e4; cases["one huge negative"] = h
halves = (np.arange(4 * 256, dtype=np.float32).reshape(4, 256) % 9 - 4) * 0.5; halves[:, 0] = 127; cases["exact halves"] = halves

bad = 0
for name, x in cases.items():
    x = np.ascontiguousarray(x)
    a, b = pack(fast, x), pack(ref, x)
    same = np.array_equal(a, b)
    bad += not same
    print(f"{name:20s} {'same bytes' if same else f'DIFFERENT ({int((a != b).sum())} bytes)'}")

x = np.ascontiguousarray(rng.standard_normal((4, 10240)).astype(np.float32))
for label, f in (("generic", ref), ("vector", fast)):
    t0 = time.perf_counter()
    for _ in range(2000):
        pack(f, x)
    print(f"{label:8s} {1e6 * (time.perf_counter() - t0) / 2000:7.1f} us per 4 x 10240 pack (this machine)")
sys.exit(1 if bad else 0)
