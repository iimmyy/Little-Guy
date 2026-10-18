#!/usr/bin/env python3
"""Run Little Guy's moods on the Mac with llama.cpp (what export_mood.py prepares on AWS).

  bake   <src.gguf> <neurons_off.json> <out.gguf>   copy the brain and zero the switched-off neurons in the copy
         (blk.L.ffn_down columns, Q4_0: the 4-bit code 8 is exactly 0). The source is only read, never written.
  cvec   <trait_unit.npy> <first room> <last room> <out.gguf>   one natural step of the mood as a llama.cpp control
         vector; load it with --control-vector-scaled <out.gguf>:<steps> (steps = the drip dose)
  verify <llama-server> <plain.gguf> <off.gguf> <cvec.gguf> <reference.json>   run the three brains of the reference
         sheet on the Mac and compare first-word probabilities and greedy words with AWS
Needs gguf-py from the llama.cpp tree on PYTHONPATH.
"""
import json, shutil, subprocess, sys, time, urllib.request
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.quants import dequantize


def bake(src, offp, dst):
    off = {int(k): v for k, v in json.load(open(offp)).items()}
    shutil.copyfile(src, dst)
    r = GGUFReader(dst, "r+")
    by = {t.name: t for t in r.tensors}
    for L, ix in sorted(off.items()):
        t = by[f"blk.{L}.ffn_down.weight"]
        assert t.tensor_type == GGMLQuantizationType.Q4_0, t.tensor_type
        d = t.data                                               # [rows, bytes]: 18-byte blocks of 32 inputs
        for i in ix:
            b, j = divmod(i, 32)
            col = b * 18 + 2 + j % 16                            # 2 bytes of scale, then 16 bytes of nibble pairs
            d[:, col] = (d[:, col] & 0xF0) | 0x08 if j < 16 else (d[:, col] & 0x0F) | 0x80
    r.data.flush()
    # check: the switched-off columns are now exactly 0 and everything else is untouched
    a, b = {t.name: t for t in GGUFReader(src).tensors}, {t.name: t for t in GGUFReader(dst).tensors}
    for L, ix in sorted(off.items()):
        n = f"blk.{L}.ffn_down.weight"
        x = dequantize(a[n].data, a[n].tensor_type); y = dequantize(b[n].data, b[n].tensor_type)
        keep = np.ones(x.shape[1], bool); keep[ix] = False
        assert np.all(y[:, ix] == 0) and np.array_equal(x[:, keep], y[:, keep]), n
    changed = [n for n in a if n not in {f"blk.{L}.ffn_down.weight" for L in off}
               and not np.array_equal(np.asarray(a[n].data), np.asarray(b[n].data))]
    assert not changed, changed
    print(f"baked {sum(map(len, off.values()))} neurons off in rooms {sorted(off)} -> {dst}; all other tensors equal")


def cvec(unitp, first, last, dst):
    unit = np.load(unitp)
    w = GGUFWriter(dst, "controlvector")
    w.add_string("controlvector.model_hint", "gemma4")
    w.add_uint32("controlvector.layer_count", last - first + 1)
    for L in range(int(first), int(last) + 1):
        w.add_tensor(f"direction.{L}", unit[L].astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"control vector rooms {first}-{last}, size per room {np.linalg.norm(unit[int(first)]):.1f}"
          f"..{np.linalg.norm(unit[int(last)]):.1f} -> {dst}")


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def serve(server, model, extra, port=8191):
    p = subprocess.Popen([server, "-m", model, "--port", str(port), "-c", "2048", "-ngl", "99", "--no-warmup",
                          *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); return p
        except Exception:
            time.sleep(0.5)
    p.kill(); raise RuntimeError("server did not start")


def verify(server, plain, offg, cv, refp):
    ref = json.load(open(refp))
    lo, hi = ref["drip_rooms"]
    brains = {"plain": (plain, []), "off": (offg, []),
              "off_drip": (offg, ["--control-vector-scaled", f"{cv}:{ref['drip']}",
                                  "--control-vector-layer-range", str(lo), str(hi)])}
    report = {}
    for name, (model, extra) in brains.items():
        p = serve(server, model, extra)
        try:
            for k, ids in enumerate(ref["prompts"]):
                o = post(8191, {"prompt": ids, "n_predict": 30, "temperature": 0, "n_probs": 10,
                                "cache_prompt": False, "samplers": ["top_k"], "top_k": 1})
                first = o["completion_probabilities"][0]["top_logprobs"]
                mac = {t["token"]: float(np.exp(t["logprob"])) for t in first}
                aws = ref["brains"][name][k]
                top_aws = aws["top"][0][0]
                g_mac = [c["id"] for c in o["completion_probabilities"]]
                same = next((i for i, (x, y) in enumerate(zip(g_mac, aws["greedy_ids"])) if x != y), len(g_mac))
                row = {"aws_top": [aws["top"][0][0], aws["top"][0][2]],
                       "mac_p_of_aws_top": round(mac.get(top_aws, 0.0), 4),
                       "mac_top": max(mac, key=mac.get), "same_greedy_words": f"{same}/30",
                       "aws": aws["greedy"][:60], "mac": o["content"][:60]}
                report.setdefault(name, []).append(row)
                print(name, k, json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            p.kill(); p.wait()
    return report


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "bake":
        bake(*a)
    elif cmd == "cvec":
        cvec(a[0], int(a[1]), int(a[2]), a[3])
    elif cmd == "verify":
        out = verify(*a[:5])
        if len(a) > 5:
            json.dump(out, open(a[5], "w"), indent=1, ensure_ascii=False)
