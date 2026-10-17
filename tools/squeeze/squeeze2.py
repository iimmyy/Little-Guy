"""Squeeze v2: any model in, smaller and still as smart out, in stock llama.cpp types only.

Every weight matrix of every layer gets its own type. Which type is decided by a harm model: the importance-weighted
rounding error of each matrix in each donor type (cheap, no inference), scaled per (group, depth block) by a few
measured KL probes (calibration). A budget is then spent where it removes the most predicted harm per byte.

  squeeze2.py <config> discover          list the model's weight matrices -> work/units.json
  squeeze2.py <config> harm              importance-weighted rounding error per matrix per donor type -> work/harm.json
  squeeze2.py <config> calibrate         measured KL per (group, depth block) downgrade -> work/cal.json
  squeeze2.py <config> single            measured KL of each matrix moved on its own -> work/single.json
                                         (config cost=measured makes allocate use it instead of harm x scale)
  squeeze2.py <config> allocate GB       cheapest predicted harm at GB -> work/recipe-<GB>.json
  squeeze2.py <config> build GB          assemble the recipe from the donors -> work/<name>-<GB>.gguf
  squeeze2.py <config> score FILE        KL and same-top vs the reference on the eval text(s)
  squeeze2.py <config> curve GB [GB..]   allocate + build + score at each size -> work/curve.json

Config (example tools/squeeze/gemma4-31b-v2.json): bf16 (reference weights), imatrix, donors {type: file} (whole
model at one type, --pure, same importance matrix), top (the donor everything starts from, e.g. the official 4-bit),
evals (text, KL base logits from the 8-bit, chunks), work, engine_bin, gguf_py, threads.
"""
import json, os, re, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

cfg_path, cmd, args = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
C = json.loads(cfg_path.read_text())
ROOT = Path(__file__).resolve().parents[2]


def _abs(x):
    return str(x if Path(x).is_absolute() else ROOT / x)


for k in ("work", "engine_bin", "gguf_py", "bf16", "imatrix"):
    C[k] = _abs(C[k])
C["donors"] = {t: _abs(f) for t, f in C["donors"].items()}
for ev in C["evals"] + ([C["probe"]] if "probe" in C else []):
    ev["text"], ev["base"] = _abs(ev["text"]), _abs(ev["base"])
W = Path(C["work"]); W.mkdir(parents=True, exist_ok=True); B = Path(C["engine_bin"])
sys.path.insert(0, C["gguf_py"])
import gguf
from gguf.quants import dequantize

GPU = ["-ngl", "99"] if C.get("gpu") else []
UNIT = re.compile(r"blk\.(\d+)\.([a-z0-9_]+)\.weight$")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tensors(path):
    return {t.name: t for t in gguf.GGUFReader(path).tensors}


# ---------------------------------------------------------------- discover

def discover():
    """every 2D per-layer weight matrix of the model, from its own file"""
    ref = gguf.GGUFReader(C["bf16"])
    arch = ref.fields[gguf.Keys.General.ARCHITECTURE].contents()
    units = {}
    for t in ref.tensors:
        m = UNIT.match(t.name)
        if m and len(t.shape) == 2:
            units[t.name] = {"layer": int(m.group(1)), "group": m.group(2), "shape": [int(x) for x in t.shape]}
    n_layer = 1 + max(u["layer"] for u in units.values())
    groups = sorted({u["group"] for u in units.values()})
    (W / "units.json").write_text(json.dumps({"arch": arch, "n_layer": n_layer, "groups": groups, "units": units}, indent=1))
    log(f"{arch}: {n_layer} layers, {len(units)} matrices, groups {groups}")


def units():
    return json.loads((W / "units.json").read_text())


# ---------------------------------------------------------------- harm

_R = {}


def _open():
    _R["ref"] = tensors(C["bf16"])
    _R["imp"] = tensors(C["imatrix"])
    _R["don"] = {ty: tensors(f) for ty, f in C["donors"].items()}


def _harm_one(name):
    """sum over sampled rows of sum_j imp_j (Wq - W)_ij^2, scaled to all rows: the expected squared output error of
    this matrix on the calibration activations, if the inputs were uncorrelated"""
    step = C.get("harm_row_step", 8)
    imp = np.asarray(_R["imp"][name + ".in_sum2"].data, dtype=np.float64) / float(_R["imp"][name + ".counts"].data[0])
    rt = _R["ref"][name]
    rows = np.arange(0, rt.data.shape[0], step)
    ref = dequantize(rt.data[rows], rt.tensor_type).astype(np.float64)
    out = {}
    for ty, d in _R["don"].items():
        t = d[name]
        q = dequantize(t.data[rows], t.tensor_type).astype(np.float64)
        out[ty] = {"harm": float(((q - ref) ** 2 @ imp) .sum() * rt.data.shape[0] / len(rows)),
                   "bytes": int(t.n_bytes), "type": t.tensor_type.name}
    return name, out


def harm():
    U = units()["units"]
    hf = W / "harm.json"
    H = json.loads(hf.read_text()) if hf.exists() else {}
    todo = [n for n in U if n not in H]
    log(f"harm: {len(todo)} matrices x {len(C['donors'])} types, row step {C.get('harm_row_step', 8)}")
    with Pool(C.get("threads", 16), initializer=_open) as p:
        for i, (name, out) in enumerate(p.imap_unordered(_harm_one, todo), 1):
            H[name] = out
            if i % 20 == 0 or i == len(todo):
                hf.write_text(json.dumps(H, indent=1)); log(f"  {i}/{len(todo)}")


def harm_from():
    """harm.json from match.py sidecars: the output error LEFT AFTER matching, per matrix per type (v3: match first,
    then allocate); only matrices matched in every type take part in allocation"""
    H = {}
    for ty, p in C["harm_from"].items():
        side = json.loads(Path(_abs(p)).read_text())
        for n, v in side.get("matrices", side).items():   # held-out error when the sidecar has it (fresh builds)
            H.setdefault(n, {})[ty] = {"harm": v.get("hold_matched", v["matched"]), "bytes": v["bytes"], "type": v["type"]}
    H = {n: o for n, o in H.items() if len(o) == len(C["harm_from"])}
    (W / "harm.json").write_text(json.dumps(H, indent=1))
    log(f"harm from matching: {len(H)} matrices x {len(C['harm_from'])} types")


# ---------------------------------------------------------------- build and score

def assemble(recipe, out, align=None):
    """recipe {matrix name: type}; every other tensor and all metadata come from the top donor. align (power of two)
    starts every tensor on that boundary, so a matrix whose types all fit in one block can change type in place."""
    top = gguf.GGUFReader(C["donors"][C["top"]])
    by = {ty: tensors(f) for ty, f in C["donors"].items() if ty in set(recipe.values())}
    w = gguf.GGUFWriter(str(out), top.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=top.endianess)
    if align:
        w.add_custom_alignment(align)
    for f in top.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    ts = [by[recipe[t.name]][t.name] if t.name in recipe else t for t in top.tensors]
    for t in ts:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in ts:
        w.write_tensor_data(t.data, tensor_endianess=top.endianess)
    w.close()
    return sum(int(t.n_bytes) for t in ts)


def score(path, evals=None, chunks=None):
    res = {}
    for ev in (evals or C["evals"]):
        env = dict(os.environ, LG_DICT_PREVIEW="0", **ev.get("env", {}))
        r = subprocess.run([str(B / "llama-perplexity"), "-m", str(path), "-f", ev["text"], "-c", str(ev.get("ctx", 512)),
                            "--chunks", str(chunks or ev["chunks"]), "-t", str(C.get("threads", 16))] + GPU + [
                            "--kl-divergence-base", ev["base"], "--kl-divergence"], capture_output=True, text=True, env=env)
        o = r.stdout + r.stderr
        kld = re.search(r"Mean\s+KLD:\s+([0-9.]+)\s+±\s+([0-9.]+)", o)
        top = re.search(r"Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)", o)
        res[ev["name"]] = {"kld": float(kld.group(1)) if kld else None, "kld_se": float(kld.group(2)) if kld else None,
                           "top": float(top.group(1)) if top else None, "top_se": float(top.group(2)) if top else None}
    return res


# ---------------------------------------------------------------- paired scoring

def read_logprobs(path):
    """llama-perplexity --kl-divergence-base file -> (per-token log-prob rows, n_vocab, tokens per chunk)"""
    with open(path, "rb") as f:
        assert f.read(8) == b"_logits_"
        n_ctx, n_vocab, n_chunk = (int(v) for v in np.frombuffer(f.read(12), dtype=np.int32))   # Python ints: offsets pass 2^31
    nv = 2 * ((n_vocab + 1) // 2) + 4
    off = 8 + 12 + 4 * n_chunk * n_ctx
    rows = np.memmap(path, dtype=np.uint16, mode="r", offset=off).reshape(-1, nv)
    return rows, int(n_vocab), int(n_ctx - 1 - n_ctx // 2)


def _decode(rows, n_vocab):
    head = np.ascontiguousarray(rows[:, :4]).view(np.float32)       # [scale, min log-prob] per token
    return head[:, 1:2].astype(np.float64) + rows[:, 4:4 + n_vocab].astype(np.float64) * head[:, 0:1]


def token_kl(ref_path, cand_path, step=32):
    """per-token KL(reference || candidate) from two saved log-prob files of the same text"""
    ref, n_vocab, _ = read_logprobs(ref_path)
    cand, n2, _ = read_logprobs(cand_path)
    assert n_vocab == n2 and ref.shape == cand.shape
    out = np.empty(ref.shape[0])
    try:   # same sum on a GPU when there is one (float64 there too)
        import torch
        dev = "cuda" if torch.cuda.is_available() else None
    except ImportError:
        dev = None
    if dev:
        def dec(rows):
            r = torch.from_numpy(np.ascontiguousarray(rows).view(np.int16)).to(dev).to(torch.int32) & 0xFFFF
            head = torch.from_numpy(np.ascontiguousarray(rows[:, :4]).view(np.float32).copy()).to(dev).double()
            return head[:, 1:2] + r[:, 4:4 + n_vocab].double() * head[:, 0:1]
        step = 512
        for i in range(0, ref.shape[0], step):
            lr, lc = dec(ref[i:i + step]), dec(cand[i:i + step])
            out[i:i + step] = (torch.exp(lr) * (lr - lc)).sum(1).cpu().numpy()
        return out
    for i in range(0, ref.shape[0], step):
        lr, lc = _decode(ref[i:i + step], n_vocab), _decode(cand[i:i + step], n_vocab)
        out[i:i + step] = (np.exp(lr) * (lr - lc)).sum(1)
    return out


TMP = Path(os.environ.get("LG_SQ_TMP", str(W)))   # e.g. /dev/shm: probe builds and candidate logits stay in RAM


def score_tokens(path, ev):
    """run the model once over the eval text, saving its log-probs; return its per-token KL vs the reference"""
    env = dict(os.environ, LG_DICT_PREVIEW="0", **ev.get("env", {}))
    if C.get("kl_dump"):   # the engine compares against the reference itself and writes only the per-token KL
        out = TMP / "cand.kld"
        env["LG_KL_DUMP"] = str(out)
        subprocess.run([str(B / "llama-perplexity"), "-m", str(path), "-f", ev["text"], "-c", str(ev.get("ctx", 512)),
                        "--chunks", str(ev["chunks"]), "-t", str(C.get("threads", 16))] + GPU + [
                        "--kl-divergence-base", ev["base"], "--kl-divergence"], capture_output=True, text=True, env=env, check=True)
        k = np.fromfile(out, dtype=np.float32).astype(np.float64)
        out.unlink()
        return k
    tmp = TMP / "cand.kl"
    subprocess.run([str(B / "llama-perplexity"), "-m", str(path), "-f", ev["text"], "-c", str(ev.get("ctx", 512)),
                    "--chunks", str(ev["chunks"]), "-t", str(C.get("threads", 16))] + GPU + ["--kl-divergence-base", str(tmp)],
                   capture_output=True, text=True, env=env, check=True)
    k = token_kl(ev["base"], tmp)
    tmp.unlink()
    return k


def paired(k, k0, per_chunk):
    """mean KL increase over the base on the same tokens; standard error from chunk means (tokens within a chunk
    are correlated, chunks are not)"""
    d = (k - k0).reshape(-1, per_chunk).mean(1)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


# ---------------------------------------------------------------- calibrate

def depth_block(layer, n_layer):
    return layer * C.get("cal_blocks", 4) // n_layer


def calibrate():
    """measured KL of moving one (group, depth block) from the top type to cal_type, on top of all-top;
    scale = measured KL increase / predicted harm increase for those matrices"""
    D = units(); U = D["units"]; H = json.loads((W / "harm.json").read_text())
    top, low = C["top"], C["cal_type"]
    cf = W / "cal.json"
    cal = json.loads(cf.read_text()) if cf.exists() else {}
    tmp = W / "cal.gguf"
    ev = C["probe"]   # its own dev text: never one of the rulers results are reported on
    assert all(ev["text"] != e["text"] for e in C["evals"]), "probe text must not be a ruler"
    _, _, per_chunk = read_logprobs(ev["base"])
    bf = W / "base_tokkl.npy"
    if not bf.exists():
        assemble({}, tmp)
        np.save(bf, score_tokens(tmp, ev))
    k0 = np.load(bf)
    cal["base"] = {"kld": float(k0.mean()), "tokens": len(k0)}
    log(f"base: KL {k0.mean():.4f} over {len(k0)} tokens")
    for g in sorted({U[n]["group"] for n in H}):
        for b in range(C.get("cal_blocks", 4)):
            key = f"{g}:{b}"
            if key in cal and "dkl_se" in cal[key]:
                continue
            names = [n for n, u in U.items() if n in H and u["group"] == g and depth_block(u["layer"], D["n_layer"]) == b]
            assemble({n: low for n in names}, tmp)
            dk, se = paired(score_tokens(tmp, ev), k0, per_chunk)
            dh = sum(H[n][low]["harm"] - H[n][top]["harm"] for n in names)
            cal[key] = {"dkl": dk, "dkl_se": se, "dharm": dh, "scale": dk / dh if dh > 0 else None}
            cf.write_text(json.dumps(cal, indent=1))
            log(f"{key} -> {low}: KL +{dk:.4f} ± {se:.4f} (paired), harm +{dh:.3g}, scale {cal[key]['scale']}")
    tmp.unlink(missing_ok=True)


def single():
    """measured, not predicted: move ONE matrix from the top type to each lower type on top of the start file, measure
    the paired KL increase on the probe text, put it back. First every matrix at the first lower type; a further type
    only for matrices whose cost at the previous type is below single_next_below (a matrix that already hurts at 3
    bits will not be picked at 2). Resumes from work/single.json."""
    H = json.loads((W / "harm.json").read_text())
    lows = list(C["single_types"])
    sf = W / C.get("single_file", "single.json")
    S = json.loads(sf.read_text()) if sf.exists() else {}
    ev = C["probe"]
    assert all(ev["text"] != e["text"] for e in C["evals"]), "probe text must not be a ruler"
    _, _, per_chunk = read_logprobs(ev["base"])
    bf = W / ("base_tokkl_dump.npy" if C.get("kl_dump") else "base_tokkl.npy")
    if not bf.exists():
        assemble({}, TMP / "probe.gguf"); np.save(bf, score_tokens(TMP / "probe.gguf", ev))
    k0 = np.load(bf)
    probe = TMP / "probe.gguf"
    if C.get("inplace"):
        # one copy of the start file, rebuilt fresh each run (a crash can't leave a swapped matrix behind); each probe
        # writes one matrix's lower-type bytes into its own slot (lower types are smaller) and its type number into the
        # tensor table, runs, and restores both. Same bytes as a full rebuild, without writing 5 GB per probe.
        A = C.get("inplace_align", 1 << 24)   # 16 MiB: every E4B block matrix fits one block at any type
        assemble({}, probe, align=A)
        import hashlib
        def digest():
            h = hashlib.sha256()
            with open(probe, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            return h.hexdigest()
        start_hash = digest()
        log(f"in-place probe copy {start_hash[:12]}")
        R = gguf.GGUFReader(str(probe), "r+")
        T = {t.name: t for t in R.tensors}
        donor = {ty: tensors(C["donors"][ty]) for ty in lows}

        def put(n, ty):
            t, src = T[n], donor[ty][n]
            new = np.asarray(src.data).view(np.uint8).ravel()
            pad = lambda x: (x + A - 1) // A * A
            assert tuple(src.shape) == tuple(t.shape) and pad(new.nbytes) == pad(int(t.n_bytes)), n
            old = (int(t.field.parts[4][0]), np.array(R.data[t.data_offset:t.data_offset + int(t.n_bytes)]))
            R.data[t.data_offset:t.data_offset + new.nbytes] = new
            t.field.parts[4][0] = int(src.tensor_type)
            R.data.flush()
            return old

        def take_back(n, old):
            t = T[n]
            R.data[t.data_offset:t.data_offset + old[1].nbytes] = old[1]
            t.field.parts[4][0] = old[0]
            R.data.flush()
    for i, ty in enumerate(lows):
        names = sorted(H)
        if i > 0:
            prev, cut = lows[i - 1], C.get("single_next_below", 1e-4)
            names = [n for n in names if S.get(n, {}).get(prev, {}).get("dkl", 1) < cut]
        todo = [n for n in names if ty not in S.get(n, {})]
        log(f"single {ty}: {len(todo)} matrices to measure ({len(names)} eligible)")
        for j, n in enumerate(todo, 1):
            if C.get("inplace"):
                old = put(n, ty)
                try:
                    k = score_tokens(probe, ev)
                finally:
                    take_back(n, old)
            else:
                assemble({n: ty}, probe)
                k = score_tokens(probe, ev)
            dk, se = paired(k, k0, per_chunk)
            S.setdefault(n, {})[ty] = {"dkl": dk, "dkl_se": se, "saved": H[n][C["top"]]["bytes"] - H[n][ty]["bytes"]}
            sf.write_text(json.dumps(S, indent=1))
            log(f"  {j}/{len(todo)} {n} -> {ty}: KL +{dk:.6f} +- {se:.6f}, saves {S[n][ty]['saved'] / 1e6:.1f} MB")
    if C.get("inplace"):
        end_hash = digest()
        log(f"in-place probe copy at the end {end_hash[:12]}: {'every matrix restored' if end_hash == start_hash else 'CHANGED'}")
        assert end_hash == start_hash, "probe copy changed: a matrix was not restored"
    probe.unlink(missing_ok=True)


def scales():
    """(group, block) -> KL per unit of harm; groups without a usable measurement use the median"""
    cal = json.loads((W / "cal.json").read_text())
    s = {k: v["scale"] for k, v in cal.items() if k != "base" and v.get("scale") and v["scale"] > 0}
    med = float(np.median(list(s.values())))
    return s, med


# ---------------------------------------------------------------- allocate

def allocate(gb):
    """Lagrangian knapsack: for a price lam per byte each matrix takes the type minimising predicted KL + lam * bytes;
    bisect lam until the file fits gb. Predicted KL per matrix = scale(group, block) x harm."""
    D = units(); U = D["units"]; H = json.loads((W / "harm.json").read_text())
    S, med = scales() if C.get("cost") != "measured" else ({}, 0.0)
    other = sum(int(t.n_bytes) for n, t in tensors(C["donors"][C["top"]]).items() if n not in H)
    budget = gb * 1e9 - other
    if C.get("cost") == "measured":   # single.json: each matrix's own measured KL increase per type (top costs 0)
        M = json.loads((W / C.get("single_file", "single.json")).read_text())
        cost = {n: {ty: (0.0 if ty == C["top"] else M[n][ty]["dkl"], h["bytes"]) for ty, h in H[n].items()
                    if ty == C["top"] or ty in M.get(n, {})} for n in H}
    else:
        cost = {n: {ty: (S.get(f"{U[n]['group']}:{depth_block(U[n]['layer'], D['n_layer'])}", med) * h["harm"], h["bytes"])
                    for ty, h in H[n].items()} for n in H}

    def pick(lam):
        r = {n: min(o, key=lambda ty: o[ty][0] + lam * o[ty][1]) for n, o in cost.items()}
        return r, sum(cost[n][ty][1] for n, ty in r.items()), sum(cost[n][ty][0] for n, ty in r.items())

    lo, hi = 0.0, 1.0
    while pick(hi)[1] > budget:
        hi *= 4
        if hi > 1e30:
            sys.exit(f"{gb} GB is below the smallest donor mix")
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pick(mid)[1] <= budget else (mid, hi)
    r, nb, pk = pick(hi)
    counts = {}
    for ty in r.values():
        counts[ty] = counts.get(ty, 0) + 1
    rec = {"gb": gb, "bytes": nb + other, "predicted_kl_increase": pk, "types": counts, "recipe": r}
    (W / f"recipe-{gb}.json").write_text(json.dumps(rec, indent=1))
    log(f"{gb} GB: {(nb + other) / 1e9:.3f} GB, predicted KL +{pk:.4f} over all-{C['top']}, {counts}")
    return rec


def build(gb):
    rec = json.loads((W / f"recipe-{gb}.json").read_text())
    out = W / f"{C['name']}-{gb}.gguf"
    n = assemble(rec["recipe"], out)
    log(f"built {out.name}: {n / 1e9:.3f} GB")
    return out


def curve(gbs):
    cf = W / "curve.json"
    cv = json.loads(cf.read_text()) if cf.exists() else {}
    for gb in gbs:
        rec = allocate(gb)
        out = build(gb)
        cv[str(gb)] = {"bytes": rec["bytes"], "types": rec["types"], "predicted_kl_increase": rec["predicted_kl_increase"],
                       "score": score(out)}
        cf.write_text(json.dumps(cv, indent=1)); log(f"{gb} GB scored: {cv[str(gb)]['score']}")
        if not C.get("keep_builds"):
            out.unlink()


if cmd == "discover": discover()
elif cmd == "harm": harm()
elif cmd == "harm_from": harm_from()
elif cmd == "calibrate": calibrate()
elif cmd == "single": single()
elif cmd == "allocate": allocate(float(args[0]))
elif cmd == "build": build(float(args[0]))
elif cmd == "score": print(json.dumps(score(Path(args[0])), indent=1))
elif cmd == "curve": curve([float(x) for x in args])
else: sys.exit(__doc__)
