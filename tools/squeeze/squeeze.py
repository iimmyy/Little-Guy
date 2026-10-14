"""Squeeze pipeline: find where a model tolerates fewer bits, spend a size budget where it buys the most quality,
build the result from whole-model "donor" files, and measure it. Runs on a Linux box with the engine built
(see research/big-model/README.md); one config JSON per model (example: tools/squeeze/gemma4-31b.json).

  squeeze.py <config> donors      quantize the donor levels the config lists (skips existing files)
  squeeze.py <config> probe       one variant per (group, layer block, level) on the base recipe -> probes.json
  squeeze.py <config> allocate    greedy: best KL drop per extra byte until the budget -> recipe.json
  squeeze.py <config> build       assemble recipe.json from the donors -> <name>.gguf
  squeeze.py <config> score FILE  KL and same-top vs the reference on the eval text(s)

Probes assume the gains of different (group, block) upgrades add up; the built result is always re-measured.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

cfg_path, cmd = Path(sys.argv[1]), sys.argv[2]
C = json.loads(cfg_path.read_text())
S = Path(C["scratch"]); B = Path(C["engine_bin"]); OUT = Path(C.get("work", S / "squeeze")); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, C["gguf_py"])
import gguf

LEVELS = C["levels"]                         # {"2": {"files": [...], "type": "Q2_K"}, ...}; a donor with "quantize" is made by donors
GROUPS = C["groups"]                         # ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
N_LAYER, BLOCK = C["n_layer"], C["block"]
BASE = C["base"]                             # level per group for the starting recipe, e.g. {"ffn_gate": "2", ...}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def donors():
    for d in C["donors"]:
        f = Path(d["file"])
        if f.exists():
            log(f"donor {f.name} exists"); continue
        log(f"donor {f.name}: quantizing")
        env = dict(os.environ, **d.get("env", {}))
        subprocess.run([str(B / "llama-quantize")] + d["args"] + [C["bf16"], str(f), d["ftype"], str(C.get("threads", 16))],
                       check=True, env=env, stdout=subprocess.DEVNULL)


def level_tensors():
    """level -> {tensor name: tensor}, each from the first of the level's files holding it in the level's type"""
    by = {}
    for lv, spec in LEVELS.items():
        by[lv] = {}
        for f in spec["files"]:
            for t in gguf.GGUFReader(f).tensors:
                if t.name not in by[lv] and (spec.get("type") is None or t.tensor_type.name == spec["type"]):
                    by[lv][t.name] = t
    return by


def tensor_level(name, recipe):
    """recipe: {"groups": {group: level}, "blocks": {"group:block": level}} -> level for a blk tensor, else None"""
    m = re.match(r"blk\.(\d+)\.([a-z_]+)\.weight$", name)
    if not m or m.group(2) not in GROUPS:
        return None
    il, g = int(m.group(1)), m.group(2)
    return recipe.get("blocks", {}).get(f"{g}:{il // BLOCK}", recipe["groups"].get(g))


def assemble(recipe, out):
    """write a GGUF taking each blk tensor from the donor of its level, everything else from the base donor"""
    by = level_tensors()
    first = gguf.GGUFReader(C["frame"])
    w = gguf.GGUFWriter(str(out), first.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=first.endianess)
    for f in first.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    tensors = []
    for t in first.tensors:
        lv = tensor_level(t.name, recipe)
        tensors.append(by[lv][t.name] if lv is not None else t)
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in tensors:
        w.write_tensor_data(t.data, tensor_endianess=first.endianess)
    w.close()
    return sum(int(t.n_bytes) for t in tensors)


def score(path, chunks=None, evals=None):
    res = {}
    for ev in (evals or C["evals"]):
        env = dict(os.environ, LG_DICT_PREVIEW="0", **ev.get("env", {}))
        r = subprocess.run([str(B / "llama-perplexity"), "-m", str(path), "-f", ev["text"], "-c", str(ev.get("ctx", 512)),
                            "--chunks", str(chunks or ev["chunks"]), "-t", str(C.get("threads", 16)),
                            "--kl-divergence-base", ev["base"], "--kl-divergence"], capture_output=True, text=True, env=env)
        kld = re.search(r"Mean\s+KLD:\s+([0-9.]+)", r.stdout + r.stderr)
        top = re.search(r"Same top p:\s+([0-9.]+)", r.stdout + r.stderr)
        res[ev["name"]] = {"kld": float(kld.group(1)) if kld else None, "top": float(top.group(1)) if top else None}
    return res


def kl_sum(res):
    return sum(v["kld"] for v in res.values() if v["kld"] is not None)


def probe():
    pf = OUT / "probes.json"
    P = json.loads(pf.read_text()) if pf.exists() else {}
    base = {"groups": dict(BASE)}
    tmp = OUT / "probe.gguf"
    if "base" not in P:
        P["base"] = {"bytes": assemble(base, tmp), "score": score(tmp, C["probe_chunks"], C["evals"][:1])}
        pf.write_text(json.dumps(P, indent=1)); log("base", P["base"])
    for g in GROUPS:
        for b in range((N_LAYER + BLOCK - 1) // BLOCK):
            for lv in C["probe_levels"]:
                if int(lv) <= int(BASE[g]):
                    continue
                k = f"{g}:{b}:{lv}"
                if k in P:
                    continue
                r = {"groups": dict(BASE), "blocks": {f"{g}:{b}": lv}}
                P[k] = {"bytes": assemble(r, tmp), "score": score(tmp, C["probe_chunks"], C["evals"][:1])}
                pf.write_text(json.dumps(P, indent=1))
                d = kl_sum(P["base"]["score"]) - kl_sum(P[k]["score"])
                log(f"{k}: +{(P[k]['bytes'] - P['base']['bytes']) / 1e9:.2f} GB, KL drop {d:.4f}")
    tmp.unlink(missing_ok=True)


def allocate():
    P = json.loads((OUT / "probes.json").read_text())
    base_b, base_k = P["base"]["bytes"], kl_sum(P["base"]["score"])
    budget = C["budget_gb"] * 1e9
    cand = {}
    for k, v in P.items():
        if k == "base":
            continue
        g, b, lv = k.split(":")
        cand.setdefault(f"{g}:{b}", {})[lv] = (v["bytes"] - base_b, base_k - kl_sum(v["score"]))
    chosen, used = {}, base_b
    while True:   # each step: the single best upgrade of one (group, block) by KL drop per extra byte
        best = None
        for gb, opts in cand.items():
            cur = chosen.get(gb)
            cb, ck = (opts[cur] if cur else (0, 0.0))
            for lv, (db, dk) in opts.items():
                if cur and int(lv) <= int(cur):
                    continue
                xb, xk = db - cb, dk - ck
                if xb <= 0 or xk <= 0 or used + xb > budget:
                    continue
                if best is None or xk / xb > best[0]:
                    best = (xk / xb, gb, lv, xb, xk)
        if best is None:
            break
        _, gb, lv, xb, xk = best
        chosen[gb] = lv; used += xb
        log(f"upgrade {gb} -> {lv}-bit: +{xb / 1e9:.2f} GB (total {used / 1e9:.2f}), predicted KL -{xk:.4f}")
    recipe = {"groups": dict(BASE), "blocks": chosen, "predicted_bytes": used,
              "predicted_kl": base_k - sum(cand[gb][lv][1] for gb, lv in chosen.items())}
    (OUT / "recipe.json").write_text(json.dumps(recipe, indent=1))
    log(f"recipe: {len(chosen)} upgrades, {used / 1e9:.2f} GB, predicted KL {recipe['predicted_kl']:.4f}")


def build():
    recipe = json.loads((OUT / "recipe.json").read_text())
    out = OUT / f"{C['name']}.gguf"
    n = assemble(recipe, out)
    log(f"built {out}: {n / 1e9:.2f} GB")


if cmd == "donors": donors()
elif cmd == "probe": probe()
elif cmd == "allocate": allocate()
elif cmd == "build": build()
elif cmd == "score": print(json.dumps(score(Path(sys.argv[3])), indent=1))
else: sys.exit(__doc__)
