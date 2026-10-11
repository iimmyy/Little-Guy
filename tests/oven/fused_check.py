"""Oven check: the fused few-word 2-bit recipe and the template recipe against llama.cpp's generic reference.

Random q2_Kx8 trays (sane scales) times random activations packed for 4 or 8 words. The fused path runs for word
counts up to LG_Q2K_FUSED_MAX; each recipe runs in its own process, chosen by that variable.

Usage: python fused_check.py [path to libggml-cpu]      (default: the Mac build)
"""
import ctypes, os, subprocess, sys
from pathlib import Path

import numpy as np

LIB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / ".run/lcpp-src/build-mac/bin/libggml-cpu.dylib"
K, N = 2560, 1024
TRAY = 84 * 8                                     # block_q2_Kx8: d[8], dmin[8], scales[128], qs[512]


def compute(which, nr):
    lib = ctypes.CDLL(str(LIB))
    mm = {"oven": lib.ggml_gemm_q2_K_8x4_q8_K, "reference": lib.ggml_gemm_q2_K_8x4_q8_K_generic}[which]
    mm.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    pack = lib.ggml_quantize_mat_q8_K_4x4
    pack.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    rng = np.random.default_rng(1)
    w = rng.integers(0, 256, (N // 8) * (K // 256) * TRAY, dtype=np.uint8)
    halves = w.view(np.uint16).reshape(-1, TRAY // 2)
    halves[:, :16] = np.float16(0.008).view(np.uint16)          # d and dmin of the 8 rows
    x = rng.standard_normal((nr, K)).astype(np.float32)
    act = np.zeros((nr // 4) * (K // 256) * 1168, np.uint8)
    for g in range(nr // 4):
        pack(np.ascontiguousarray(x[4 * g:4 * g + 4]).ctypes.data, act[g * (K // 256) * 1168:].ctypes.data, K)
    s = np.zeros((nr, N), np.float32)
    mm(K, s.ctypes.data, N, w.ctypes.data, act.ctypes.data, nr, N)
    return s


if __name__ == "__main__":
    if os.environ.get("FUSED_CHILD"):
        which, nr = os.environ["FUSED_CHILD"].split(",")
        np.save(sys.stdout.buffer, compute(which, int(nr)))
        sys.exit()
    import io
    bad = 0
    for nr in (4, 8):
        out = {}
        for label, which, fused_max in (("reference", "reference", "8"), ("fused", "oven", "8"), ("template", "oven", "0")):
            env = dict(os.environ, FUSED_CHILD=f"{which},{nr}", LG_Q2K_FUSED_MAX=fused_max)
            out[label] = np.load(io.BytesIO(subprocess.run([sys.executable, __file__, str(LIB)], env=env, capture_output=True, check=True).stdout))
        ref = out["reference"]
        for label in ("fused", "template"):
            err = np.abs(out[label] - ref).max() / np.abs(ref).max()
            ok = err < 1e-5
            bad += not ok
            print(f"{nr} words, {label:8s}: largest difference {err:.1e} of the largest answer  {'ok' if ok else 'WRONG'}")
    sys.exit(1 if bad else 0)
