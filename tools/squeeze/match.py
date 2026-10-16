"""Output matching for stock llama.cpp types: keep every block scale of a stock-quantized matrix exactly as it is and
re-pick only the integer codes, column by column, so the matrix's output on real inputs matches the original
(GPTQ-style error compensation with a fixed grid). The result is byte-compatible with the stock type.

  match.py test  <ref.gguf> <donor.gguf> <act dir> <tensor>   one matrix: output error, stock rounding vs matched
  match.py build <ref.gguf> <donor.gguf> <act dir> <out.gguf> every matrix with recorded inputs, matched

ref: the full-precision model (bf16). donor: the whole model at one stock type (same importance matrix as usual).
act dir: input rows recorded by llama-imatrix with LG_ACT_DUMP (one file per distinct input, f16).
Supported types: Q4_0, Q3_K, Q2_K.
"""
import json, os, re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", str(Path(__file__).resolve().parents[2] / ".run/lcpp-src/gguf-py")))
import gguf
from gguf.quants import dequantize

T = gguf.GGMLQuantizationType


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ grids: value = a * code + b, per element

def grid(raw, qt, n_col):
    """raw block bytes (rows, row bytes) -> per-element (a, b, code) arrays of shape (rows, n_col), codes in 0..qmax"""
    R = raw.shape[0]
    if qt == T.Q4_0:
        blk = raw.reshape(R, -1, 18)
        d = blk[:, :, :2].copy().view(np.float16)[:, :, 0].astype(np.float32)
        qs = blk[:, :, 2:]
        code = np.concatenate([qs & 15, qs >> 4], axis=2)                       # element j and j+16
        a = np.repeat(d, 32, axis=1)
        return a, -8 * a, code.reshape(R, n_col).astype(np.int16), 15
    if qt == T.Q2_K:
        blk = raw.reshape(R, -1, 84)
        sc, qs = blk[:, :, :16], blk[:, :, 16:80]
        d = blk[:, :, 80:82].copy().view(np.float16)[:, :, 0].astype(np.float32)
        dm = blk[:, :, 82:84].copy().view(np.float16)[:, :, 0].astype(np.float32)
        code, a, b = _k_layout(R, qs, n_col)
        idx = _k_scale_index()                                                   # (256,) -> scale slot
        a[:] = (d[:, :, None] * (sc[:, :, idx] & 15)).reshape(R, n_col)
        b[:] = (-dm[:, :, None] * (sc[:, :, idx] >> 4)).reshape(R, n_col)
        return a, b, code, 3
    if qt == T.Q3_K:
        blk = raw.reshape(R, -1, 110)
        hm, qs, scb = blk[:, :, :32], blk[:, :, 32:96], blk[:, :, 96:108]
        d = blk[:, :, 108:110].copy().view(np.float16)[:, :, 0].astype(np.float32)
        low, a, b = _k_layout(R, qs, n_col)
        hi = _k_hbits(hm).reshape(R, n_col)
        s = _q3k_scales(scb)                                                     # (R, nb, 16) in -32..31
        a[:] = (d[:, :, None] * s[:, :, _k_scale_index()]).reshape(R, n_col)
        b[:] = -4 * a
        return a, b, (low + 4 * hi).astype(np.int16), 7
    raise SystemExit(f"type {qt.name} not supported")


def _k_pos():
    """element position in a 256 super-block for (n, j, h, l): n*128 + j*32 + h*16 + l"""
    n, j, h, l = np.meshgrid(np.arange(2), np.arange(4), np.arange(2), np.arange(16), indexing="ij")
    return n, j, h, l, (n * 128 + j * 32 + h * 16 + l)


def _k_scale_index():
    n, j, h, l, pos = _k_pos()
    idx = np.empty(256, dtype=np.int64)
    idx[pos.ravel()] = (n * 8 + j * 2 + h).ravel()
    return idx


def _k_layout(R, qs, n_col):
    """2-bit low codes of Q2_K / Q3_K: element (n,j,h,l) lives in qs[n*32 + h*16 + l] at bit shift 2j"""
    n, j, h, l, pos = _k_pos()
    byte = (n * 32 + h * 16 + l).ravel(); shift = (2 * j).ravel()
    code = np.empty((R, qs.shape[1], 256), dtype=np.int16)
    code[:, :, pos.ravel()] = (qs[:, :, byte] >> shift) & 3
    return code.reshape(R, n_col), np.empty((R, n_col), np.float32), np.empty((R, n_col), np.float32)


def _k_hbits(hm):
    """Q3_K high bit of element (n,j,h,l): hmask[h*16 + l] bit (n*4 + j)"""
    n, j, h, l, pos = _k_pos()
    out = np.empty(hm.shape[:2] + (256,), dtype=np.int16)
    out[:, :, pos.ravel()] = (hm[:, :, (h * 16 + l).ravel()] >> (n * 4 + j).ravel()) & 1
    return out


def _q3k_scales(scb):
    """12 bytes -> 16 signed 6-bit scales (ggml's Q3_K packing), minus 32"""
    aux = scb.copy().view(np.uint32)                                             # (R, nb, 3)
    a0, a1, tmp = aux[..., 0], aux[..., 1], aux[..., 2]
    k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
    w = [(a0 & k2) | (((tmp >> 0) & k1) << 4), (a1 & k2) | (((tmp >> 2) & k1) << 4),
         ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)]
    return np.stack(w, axis=-1).view(np.int8).astype(np.int32) - 32              # (R, nb, 16)


def pack(raw, qt, code):
    """write new codes into a copy of the stock block bytes (scales untouched)"""
    R = raw.shape[0]
    out = raw.copy()
    if qt == T.Q4_0:
        c = code.reshape(R, -1, 32).astype(np.uint8)
        out.reshape(R, -1, 18)[:, :, 2:] = c[:, :, :16] | (c[:, :, 16:] << 4)
        return out
    n, j, h, l, pos = _k_pos()
    nb = code.shape[1] // 256
    c = code.reshape(R, nb, 256)[:, :, pos.ravel()].reshape(R, nb, 2, 4, 2, 16).astype(np.uint8)  # n j h l
    low = c & 3
    qs = np.zeros((R, nb, 2, 2, 16), np.uint8)                                   # n h l
    for jj in range(4):
        qs |= low[:, :, :, jj] << (2 * jj)
    if qt == T.Q2_K:
        out.reshape(R, nb, 84)[:, :, 16:80] = qs.reshape(R, nb, 64)
        return out
    hm = np.zeros((R, nb, 2, 16), np.uint8)                                      # h l
    for nn in range(2):
        for jj in range(4):
            hm |= ((c[:, :, nn, jj] >> 2) & 1) << (nn * 4 + jj)
    blk = out.reshape(R, nb, 110)
    blk[:, :, :32] = hm.reshape(R, nb, 32)
    blk[:, :, 32:96] = qs.reshape(R, nb, 64)
    return out


# ------------------------------------------------------------------ matching

ACT = {"attn_q": "attn_q", "attn_k": "attn_q", "attn_v": "attn_q", "ffn_gate": "ffn_gate", "ffn_up": "ffn_gate",
       "attn_output": "attn_output", "ffn_down": "ffn_down"}


def hessian(act_dir, name):
    m = re.match(r"(blk\.\d+)\.([a-z_]+)\.weight$", name)
    f = Path(act_dir) / f"{m.group(1)}.{ACT[m.group(2)]}.weight.f16"
    if not f.exists():
        return None
    x = np.fromfile(f, dtype=np.float16)
    return x, f


def gram(x, n_col):
    X = x.reshape(-1, n_col).astype(np.float32)
    return (X.T @ X) / np.float32(X.shape[0])


def out_err(E, H):
    return float(np.einsum("ij,ij->", E @ H, E))


def match(W, H, a, b, code0, qmax, block=128, damp=float(os.environ.get("LG_MATCH_DAMP", "0.01"))):
    """GPTQ with a fixed grid: W (rows, cols) float, H (cols, cols); returns new codes"""
    dt = np.float64 if W.shape[1] <= 12288 else np.float32   # 31B ffn_down (21504 inputs): float32 halves memory
    W = W.astype(dt).copy()
    H = H.astype(dt)
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H[np.diag_indices_from(H)] += damp * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T                                # upper
    a = a.astype(dt); b = b.astype(dt)
    code = code0.copy()
    C = W.shape[1]
    for i1 in range(0, C, block):
        i2 = min(i1 + block, C)
        W1 = W[:, i1:i2].copy()
        E1 = np.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for k in range(i2 - i1):
            col = i1 + k
            w, ac, bc = W1[:, k], a[:, col], b[:, col]
            ok = ac != 0
            c = code0[:, col].astype(dt)
            c[ok] = np.clip(np.rint((w[ok] - bc[ok]) / ac[ok]), 0, qmax)
            q = ac * c + bc
            e = (w - q) / Hi[k, k]
            W1[:, k:] -= np.outer(e, Hi[k, k:])
            E1[:, k] = e
            code[:, col] = c
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return code


def load(path):
    return {t.name: t for t in gguf.GGUFReader(path).tensors}


def one(ref, don, act_dir, name, check=True):
    rt, dt = ref[name], don[name]
    n_col = int(rt.shape[0])
    W = dequantize(rt.data, rt.tensor_type).reshape(-1, n_col).astype(np.float32)
    hx = hessian(act_dir, name)
    if hx is None:
        return None
    H = gram(hx[0], n_col)
    raw = np.asarray(dt.data).reshape(W.shape[0], -1)
    a, b, c0, qmax = grid(raw, dt.tensor_type, n_col)
    if check:   # our reading of the format must equal the engine's decoder exactly
        ref_vals = dequantize(dt.data, dt.tensor_type).reshape(W.shape)
        assert np.allclose(a * c0 + b, ref_vals, rtol=0, atol=1e-6 * np.abs(ref_vals).max()), f"{name}: grid decode mismatch"
    c1 = match(W, H, a, b, c0, qmax)
    new = pack(raw, dt.tensor_type, c1)
    if check:
        assert np.array_equal(pack(raw, dt.tensor_type, c0), raw), f"{name}: repack of stock codes changed bytes"
        got = dequantize(new, dt.tensor_type).reshape(W.shape)
        assert np.allclose(got, a * c1 + b, rtol=0, atol=1e-6 * np.abs(got).max()), f"{name}: packed codes mismatch"
    Ws = (a * c0 + b); Wm = (a * c1 + b)
    return new, out_err(Ws - W, H), out_err(Wm - W, H), float(np.mean(c0 != c1))


def build(ref_p, don_p, act_dir, out_p):
    ref, top = load(ref_p), gguf.GGUFReader(don_p)
    don = {t.name: t for t in top.tensors}
    w = gguf.GGUFWriter(out_p, top.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=top.endianess)
    for f in top.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    data, tot_s, tot_m, side = {}, 0.0, 0.0, {}
    names = [t.name for t in top.tensors if re.match(r"blk\.\d+\.(%s)\.weight$" % "|".join(ACT), t.name)]
    cache = os.environ.get("LG_MATCH_CACHE")                       # shards share results through a cache folder:
    shard = os.environ.get("LG_MATCH_SHARD")                       # LG_MATCH_SHARD=i/n matches every n-th matrix only
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        cf = Path(cache) / (n + ".npz") if cache else None
        if cf and cf.exists():
            z = np.load(cf); r = (z["new"], float(z["es"]), float(z["em"]), float(z["ch"]))
        elif shard and i % int(shard.split("/")[1]) != int(shard.split("/")[0]):
            continue
        else:
            r = one(ref, don, act_dir, n, check=(i < 3))
            if r is None:
                continue
            if cf:
                np.savez(cf, new=r[0], es=r[1], em=r[2], ch=r[3])
        if shard:
            log(f"{n}: output error {r[1]:.4g} -> {r[2]:.4g} ({100 * (1 - r[2] / r[1]):.1f}% less), codes changed {100 * r[3]:.1f}%")
            continue
        data[n] = r[0]; tot_s += r[1]; tot_m += r[2]
        side[n] = {"stock": r[1], "matched": r[2], "bytes": int(r[0].nbytes), "type": don[n].tensor_type.name}
        log(f"{n}: output error {r[1]:.4g} -> {r[2]:.4g} ({100 * (1 - r[2] / r[1]):.1f}% less), codes changed {100 * r[3]:.1f}%")
    if shard:
        log("shard done"); return
    for t in top.tensors:
        arr = data.get(t.name, t.data)
        w.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in top.tensors:
        w.write_tensor_data(data.get(t.name, t.data), tensor_endianess=top.endianess)
    w.close()
    Path(out_p + ".json").write_text(json.dumps(side, indent=1))   # per-matrix output error, for allocation
    log(f"built {out_p}: total output error {tot_s:.4g} -> {tot_m:.4g} ({100 * (1 - tot_m / tot_s):.1f}% less)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        ref, don = load(sys.argv[2]), load(sys.argv[3])
        t0 = time.time()
        _, es, em, ch = one(ref, don, sys.argv[4], sys.argv[5])
        log(f"{sys.argv[5]}: output error stock {es:.5g}, matched {em:.5g} ({100 * (1 - em / es):.1f}% less), "
            f"codes changed {100 * ch:.1f}%, {time.time() - t0:.1f} s")
    elif cmd == "build":
        build(*sys.argv[2:6])
    else:
        sys.exit(__doc__)
