"""Squeeze v2: any model in, smaller and still as smart out, in stock llama.cpp types only.

Every weight matrix of every layer gets its own type. Which type is decided by a harm model: the importance-weighted
rounding error of each matrix in each donor type (cheap, no inference), scaled per (group, depth block) by a few
measured KL probes (calibration). A budget is then spent where it removes the most predicted harm per byte.

  squeeze2.py <config> discover          list the model's weight matrices -> work/units.json
  squeeze2.py <config> harm              importance-weighted rounding error per matrix per donor type -> work/harm.json
  squeeze2.py <config> calibrate         measured KL per (group, depth block) downgrade -> work/cal.json
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
for ev in C["evals"]:
    ev["text"], ev["base"] = _abs(ev["text"]), _abs(ev["base"])
W = Path(C["work"]); W.mkdir(parents=True, exist_ok=True); B = Path(C["engine_bin"])
sys.path.insert(0, C["gguf_py"])
import gguf
from gguf.quants import dequantize

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


# ---------------------------------------------------------------- build and score

def assemble(recipe, out):
    """recipe {matrix name: type}; every other tensor and all metadata come from the top donor"""
    top = gguf.GGUFReader(C["donors"][C["top"]])
    by = {ty: tensors(f) for ty, f in C["donors"].items() if ty in set(recipe.values())}
    w = gguf.GGUFWriter(str(out), top.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=top.endianess)
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
                            "--chunks", str(chunks or ev["chunks"]), "-t", str(C.get("threads", 16)),
                            "--kl-divergence-base", ev["base"], "--kl-divergence"], capture_output=True, text=True, env=env)
        o = r.stdout + r.stderr
        kld = re.search(r"Mean\s+KLD:\s+([0-9.]+)\s+±\s+([0-9.]+)", o)
        top = re.search(r"Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)", o)
        res[ev["name"]] = {"kld": float(kld.group(1)) if kld else None, "kld_se": float(kld.group(2)) if kld else None,
                           "top": float(top.group(1)) if top else None, "top_se": float(top.group(2)) if top else None}
    return res


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
    ev = C["evals"][:1]
    if "base" not in cal:
        assemble({}, tmp)
        cal["base"] = score(tmp, ev); cf.write_text(json.dumps(cal, indent=1)); log("base", cal["base"])
    k0 = cal["base"][ev[0]["name"]]["kld"]
    for g in D["groups"]:
        for b in range(C.get("cal_blocks", 4)):
            key = f"{g}:{b}"
            if key in cal:
                continue
            names = [n for n, u in U.items() if u["group"] == g and depth_block(u["layer"], D["n_layer"]) == b]
            assemble({n: low for n in names}, tmp)
            s = score(tmp, ev)
            dk = s[ev[0]["name"]]["kld"] - k0
            dh = sum(H[n][low]["harm"] - H[n][top]["harm"] for n in names)
            cal[key] = {"score": s, "dkl": dk, "dharm": dh, "scale": dk / dh if dh > 0 else None}
            cf.write_text(json.dumps(cal, indent=1))
            log(f"{key} -> {low}: KL +{dk:.4f}, harm +{dh:.3g}, scale {cal[key]['scale']}")
    tmp.unlink(missing_ok=True)


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
    S, med = scales()
    other = sum(int(t.n_bytes) for n, t in tensors(C["donors"][C["top"]]).items() if n not in U)
    budget = gb * 1e9 - other
    cost = {n: {ty: (S.get(f"{U[n]['group']}:{depth_block(U[n]['layer'], D['n_layer'])}", med) * h["harm"], h["bytes"])
                for ty, h in H[n].items()} for n in U}

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
elif cmd == "calibrate": calibrate()
elif cmd == "allocate": allocate(float(args[0]))
elif cmd == "build": build(float(args[0]))
elif cmd == "score": print(json.dumps(score(Path(args[0])), indent=1))
elif cmd == "curve": curve([float(x) for x in args])
else: sys.exit(__doc__)
