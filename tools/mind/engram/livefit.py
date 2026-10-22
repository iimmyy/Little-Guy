"""Engram, live brain step 1: does the fragment finder (sae.py, trained on the raw brain in HF) still fit the thoughts
of his live brain in his real engine? The finder learned the raw brain's layer-8 stream; his live brain is the same
brain squeezed to 4 bits (grid4) and wearing his talk adapter, read by llama.cpp instead of HF.

  python livefit.py write --out DIR                         # the 64 moments (rung 0) as cases for layer-dump
  engine/kv-check/build/layer-dump BRAIN DIR/cases.jsonl DIR/NAME 8 [--lora ADAPTER] [--cpu]
  python livefit.py fit --dump DIR/NAME --sae sae_L8.pt [--ref DIR/OTHER]

fit prints how much of the memory words' stream the pieces explain (the same measure as rung0.py, which gave 0.813
on the raw brain in HF), at 64/32/16/8 pieces a word, and with --ref how far these thoughts sit from another brain's.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from moments import moments, prefix


def write(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cases.jsonl", "w") as f:
        for m in moments():
            f.write(json.dumps(dict(a=prefix(m["ctx_a"]), moment=m["text"])) + "\n")
    print(f"wrote {out / 'cases.jsonl'}")


def load(dump):
    meta = json.loads(Path(dump + ".json").read_text())
    rows = np.fromfile(dump + ".f32", dtype=np.float32).reshape(-1, meta["n_embd"])
    spans, s = [], 0
    for c in meta["cases"]:
        spans.append(rows[s + c["moment_from"]:s + c["rows"]])
        s += c["rows"]
    assert s == len(rows), f"{dump}: index says {s} rows, file has {len(rows)}"
    return meta, rows, spans


def explained(sae, h, k=None):
    with torch.no_grad():
        rt = sae.roundtrip(h, k)
    return (1 - (rt - h).pow(2).sum() / (h - h.mean(0)).pow(2).sum()).item()


def fit(a):
    from sae import load_sae
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sae = load_sae(a.sae, dev)
    meta, rows, spans = load(a.dump)
    h = torch.from_numpy(np.concatenate(spans)).to(dev)
    res = dict(dump=a.dump, adapter=meta["adapter"], layer=meta["layer"], moment_words=len(h))
    for k in (None, 32, 16, 8):
        res[f"explained_k{k or sae.k}"] = explained(sae, h, k)
    res["explained_all_words"] = explained(sae, torch.from_numpy(rows).to(dev))
    if a.ref:
        _, _, ref = load(a.ref)
        assert [len(x) for x in ref] == [len(x) for x in spans], "the two dumps read different words"
        r = torch.from_numpy(np.concatenate(ref)).to(dev)
        res["ref"] = a.ref
        res["moved_vs_ref"] = ((h - r).norm(dim=1) / r.norm(dim=1)).mean().item()   # 0 = same thoughts
        res["cos_vs_ref"] = torch.nn.functional.cosine_similarity(h, r, dim=1).mean().item()
    print(json.dumps(res, indent=1))
    Path(a.dump + ".fit.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--dump", required=True)
    f.add_argument("--sae", required=True)
    f.add_argument("--ref", default="")
    a = ap.parse_args()
    {"write": write, "fit": fit}[a.cmd](a)
