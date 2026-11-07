#!/usr/bin/env python3
"""Independent check of a baked build (Immy 2026-10-29: "a bug in the placement could skew everything"). Separate
code from the bake scripts, reading only the saved file: no self-report is trusted.
  diff       every tensor of the build against the untouched model: which layers / neuron slots / attention head
             dimensions / output rows changed, and by how much. Compared with what the build's json files say was
             placed; anything changed that no json names (or named but unchanged) is flagged
  overlap    the stream directions the placed neurons write (output column through the post-FFN norm weight): the
             cosine between every pair from different feelings (time / belief / stuck); should be ~0
  drift      the build vs untouched on the same text (plain answers, thinking and reply), mean KL of the next-word
             distributions: how much it changed everywhere else

usage: verify_build.py <untouched hf folder> <build folder> <answers jsonl> [n=12 dev=mps]
"""
import json, os, sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

base, build, ans = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
N, DEV = int(opt.get("n", 12)), opt.get("dev", "mps")
A = load_file(os.path.join(base, "model.safetensors"))
B = load_file(os.path.join(build, "model.safetensors"))
lines = [f"independent check of {build}"]

# ---- what the build says it placed
said = {}                                                  # (layer, slot) -> feeling
for fn, feel in (("urgency.json", "time"), ("belief.json", "belief"), ("stuck.json", "stuck")):
    pth = os.path.join(build, fn)
    if not os.path.exists(pth):
        continue
    j = json.load(open(pth))
    if feel == "time":
        said[(j["layer"], j["slot"])] = "time"
        for l, s in j.get("relays", []):
            said[(l, s)] = "time"
    elif feel == "belief":
        said[tuple(j["source"])] = "belief"
        for l, s in j["relays"]:
            said[(l, s)] = "belief"
    else:
        said[(j["layer"], j["stuck"])] = "stuck"; said[(j["layer"], j["guard"])] = "guard"
dups = len(said) - len({k for k in said})
lines.append(f"jsons name {len(said)} neuron slots: " + ", ".join(f"{f} {sum(v == f for v in said.values())}" for f in ("time", "belief", "stuck", "guard")))

# ---- diff
key = lambda k: k.replace("model.", "", 1)
only_a = sorted(set(A) - set(B)); only_b = sorted(set(B) - set(A))
lines.append(f"tensors only in untouched: {only_a[:4]}{'...' if len(only_a) > 4 else ''}; only in build: {only_b[:4]}")
changed_slots, other = {}, []
for k in sorted(set(A) & set(B)):
    a, b = A[k].float(), B[k].float()
    if a.shape != b.shape:
        other.append(f"{k}: shape {tuple(a.shape)} -> {tuple(b.shape)}"); continue
    d = (a - b).abs()
    if float(d.max()) == 0:
        continue
    if ".feed_forward." in k and k.endswith(("w1.weight", "w3.weight", "w2.weight")):
        l = int(k.split("layers.")[1].split(".")[0])
        sl = d.amax(1) if not k.endswith("w2.weight") else d.amax(0)
        for s in torch.nonzero(sl > 0)[:, 0].tolist():
            changed_slots.setdefault((l, s), set()).add(k.rsplit(".", 2)[-2])
    elif "self_attn" in k:
        rows_ = torch.nonzero(d.amax(1) > 0)[:, 0].tolist() if d.dim() == 2 else []
        other.append(f"{k}: {len(rows_)} rows changed (heads {sorted(set(r // 128 for r in rows_))})")
    else:
        rows_ = torch.nonzero(d.amax(-1) > 0)[:, 0].tolist() if d.dim() == 2 else []
        other.append(f"{k}: {len(rows_)} rows changed" + (f" ({rows_[:20]})" if len(rows_) <= 20 else ""))
if "lm_head.weight" in only_b:                            # untied: compare the output rows with the tied embedding
    E = A[[k for k in A if k.endswith("embed_tokens.weight")][0]].float()
    H = B["lm_head.weight"].float()
    rows_ = torch.nonzero((H - E).abs().amax(1) > 0)[:, 0].tolist()
    lines.append(f"output rows different from the untouched (tied) rows: {len(rows_)}: {rows_[:30]}")
lines.append(f"MLP neuron slots changed: {len(changed_slots)}")
extra = sorted(set(changed_slots) - set(said)); missing = sorted(set(said) - set(changed_slots))
lines.append(f"  changed but not named in any json: {extra}")
lines.append(f"  named but not changed: {missing}")
part = sorted(k for k, v in changed_slots.items() if v != {"w1", "w2", "w3"})
lines.append(f"  slots where not all of w1/w3/w2 changed: {part}")
lines += ["other tensors changed:"] + [f"  {o}" for o in other]

# ---- overlap of written directions
cols = {}
for (l, s), f in said.items():
    w2 = B[f"model.layers.{l}.feed_forward.w2.weight"].float()[:, s]
    g = B[f"model.layers.{l}.post_ffn_norm.weight"].float()
    v = w2 * g
    cols[(l, s)] = (f, v / (v.norm() + 1e-9))
ks = list(cols)
worst = []
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        fi, vi = cols[ks[i]]; fj, vj = cols[ks[j]]
        if fi != fj:
            worst.append((abs(float(vi @ vj)), ks[i], fi, ks[j], fj))
worst.sort(reverse=True)
lines.append("largest |cos| between directions of different feelings: " +
             "; ".join(f"{c:.3f} {a}{fa}-{b}{fb}" for c, a, fa, b, fb in worst[:5]))

# ---- drift on the same text
pa, pb = _lfm3(DEV, base), _lfm3(DEV, build)
tok = pa.tok
rows = [r for r in map(json.loads, open(ans)) if r["thinking"] and r["finish"] == "stop"][:N]
kls, kth, krp = [], [], []
with torch.no_grad():
    for r in rows:
        pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + "\n\nEnd with a line: Answer: <your answer>"}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
        th = tok("\n" + r["thinking"], add_special_tokens=False).input_ids[:2500]
        rp = tok("\n</think>\n\n" + r["reply"], add_special_tokens=False).input_ids[:400]
        ids = torch.tensor([pr + th + rp], device=DEV)
        la = torch.log_softmax(pa.model(input_ids=ids).logits[0].float(), -1)
        lb = torch.log_softmax(pb.model(input_ids=ids).logits[0].float(), -1)
        kl = (la.exp() * (la - lb)).sum(-1).cpu()
        kth.append(kl[len(pr):len(pr) + len(th)]); krp.append(kl[len(pr) + len(th):])
kth, krp = torch.cat(kth), torch.cat(krp)
lines.append(f"drift (KL, untouched vs build, {len(rows)} answers): thinking mean {float(kth.mean()):.4f} "
             f"(99% {float(kth.quantile(.99)):.3f}, max {float(kth.max()):.2f}); reply mean {float(krp.mean()):.4f} "
             f"(99% {float(krp.quantile(.99)):.3f}, max {float(krp.max()):.2f})")
print("\n".join(lines))
open(os.path.join(build, "verify_report.txt"), "w").write("\n".join(lines) + "\n")
print("VERIFY_DONE")
