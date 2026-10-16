"""Layer-by-layer output matching (sequential): stage by stage, record the inputs each layer really receives from the
already-compressed layers before it, and match that layer so its output equals the ORIGINAL model's output given those
drifted inputs (match.py with LG_SEQ_ACT). Earlier stages' errors are then cancelled downstream instead of piling up.

usage: seq_match.py <config.json>
config: ref (full precision; also the stand-in for the not-yet-compressed layers), recipe (squeeze2 recipe json:
matrix -> type), donors {type: stock file}, frame (file for every tensor outside the recipe), act (inputs recorded from
the original), record {ctx, chunks, every, max}: EXACTLY the settings act was recorded with, so the rows line up),
calib (text), work, out, n_layer, stages, engine_bin, workers, ngl
"""
import json, os, re, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import match as M
import gguf

C = json.loads(Path(sys.argv[1]).read_text())
W = Path(C["work"]); W.mkdir(parents=True, exist_ok=True); CACHE = W / "cache"; CACHE.mkdir(exist_ok=True)
RECIPE = json.loads(Path(C["recipe"]).read_text())["recipe"]
LAYER = re.compile(r"blk\.(\d+)\.")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def layer(n):
    m = LAYER.match(n)
    return int(m.group(1)) if m else -1


def write(frame_path, out, replace):
    """GGUF from frame_path with tensors replaced: replace {name: (raw uint8 rows, quant type)}"""
    top = gguf.GGUFReader(frame_path)
    w = gguf.GGUFWriter(str(out), top.fields[gguf.Keys.General.ARCHITECTURE].contents(), endianess=top.endianess)
    for f in top.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    data = {}
    for t in top.tensors:
        if t.name in replace:
            arr, qt = replace[t.name]
            w.add_tensor_info(t.name, arr.shape, arr.dtype, arr.nbytes, qt)   # raw block bytes, like the donors
            data[t.name] = arr
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in top.tensors:
        w.write_tensor_data(data.get(t.name, t.data), tensor_endianess=top.endianess)
    w.close()


def done_tensors():
    out = {}
    for n, ty in RECIPE.items():
        f = CACHE / f"{n}.npz"
        if f.exists():
            out[n] = (np.load(f)["new"], getattr(gguf.GGMLQuantizationType, ty))
    return out


_R = {}


def _init():
    _R["ref"] = M.load(C["ref"]); _R["don"] = {ty: M.load(f) for ty, f in C["donors"].items()}


def _job(args):
    name, seq = args
    os.environ["LG_SEQ_ACT"] = seq
    don = {name: _R["don"][RECIPE[name]][name]}
    r = M.one(_R["ref"], don, C["act"], name)
    if r is None:
        return name, None
    np.savez(CACHE / f"{name}.npz", new=r[0], rep=json.dumps(r[1]))
    return name, r[1]


def main():
    B = Path(C["engine_bin"]); n_layer, S = C["n_layer"], C["stages"]
    bounds = [round(i * n_layer / S) for i in range(S + 1)]
    for k in range(S):
        lo, hi = bounds[k], bounds[k + 1]
        todo = [n for n in RECIPE if lo <= layer(n) < hi and not (CACHE / f"{n}.npz").exists()]
        if not todo:
            continue
        seq = W / f"act_s{k}"
        if not (seq / ".done").exists():
            model = W / "stage.gguf"
            write(C["ref"], model, done_tensors())                              # earlier stages compressed, rest original
            seq.mkdir(exist_ok=True)
            log(f"stage {k}: layers {lo}-{hi - 1}, recording inputs ({len(done_tensors())} matrices already compressed)")
            R = C["record"]
            env = dict(os.environ, LG_ACT_DUMP=str(seq), LG_ACT_MAX=str(R["max"]), LG_ACT_EVERY=str(R["every"]),
                       LG_ACT_LAYERS=f"{lo}-{hi - 1}")
            subprocess.run([str(B / "llama-imatrix"), "-m", str(model), "-f", C["calib"], "--parse-special", "-o",
                            str(W / "stage.imatrix"), "-c", str(R["ctx"]), "--chunks", str(R["chunks"]),
                            "-ngl", str(C.get("ngl", 99)), "-t", "8"], env=env, check=True, capture_output=True)
            (seq / ".done").touch()
        log(f"stage {k}: matching {len(todo)} matrices, settings {M.settings()}")
        with Pool(C.get("workers", 3), initializer=_init) as p:
            for name, r in p.imap_unordered(_job, [(n, str(seq)) for n in todo]):
                if r:
                    log("  " + M.line(name, r))
    write(C["frame"], C["out"], done_tensors())
    log(f"built {C['out']}")


if __name__ == "__main__":
    main()
