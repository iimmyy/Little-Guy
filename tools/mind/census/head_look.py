#!/usr/bin/env python3
"""Where does one head look, and does it write? (imager piece 1 prep, Immy 2026-11-02)

LFM3 heads have two knobs: the query (where the head looks) and a sigmoid gate per head (whether it writes). For one
head, on word problems with scene maps, untouched model:
  cue:    "Count of <thing> (<owner>):" added where the thinking starts; at its last word: attention on the token(s) of
          the right written number, on the other written numbers, and the gate.
  period: every sentence end in the story (no cue): attention on that sentence's numbers, on all numbers, and the gate.
  other:  every other story word: the gate (baseline).
Also saves the head's query (after its norm, before position rotation) at each site, for the edit direction.

usage: head_look.py <problems.jsonl> <out.txt> [layer=14] [head=15] [hf=] [dev=mps]
"""
import json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import stated, TAIL

src, out = sys.argv[1], sys.argv[2]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
LAYER, HEAD, dev = int(opt.get("layer", 14)), int(opt.get("head", 15)), opt.get("dev", "mps")
p = _lfm3(dev, opt.get("hf"))
tok, m = p.tok, p.model
m.config._attn_implementation = "eager"          # so the layer hands back its attention weights
att = m.model.layers[LAYER].self_attn
hd = att.head_dim
got = {}
att.register_forward_hook(lambda mod, a, o: got.__setitem__("w", o[1][0, HEAD].float()))
att.gate_proj.register_forward_hook(lambda mod, a, o: got.__setitem__("g", torch.sigmoid(o[0, :, HEAD].float())))
att.q_norm.register_forward_hook(lambda mod, a, o: got.__setitem__("q", o[0, :, HEAD].float()))


def run(ids):
    with torch.no_grad():
        m.model(input_ids=torch.tensor([ids], device=dev))
    return got["w"].cpu().numpy(), got["g"].cpu().numpy(), got["q"].cpu().numpy()


rows = list(map(json.loads, open(src)))
R = {"cue_right": [], "cue_other": [], "cue_gate": [], "per_own": [], "per_all": [], "per_gate": [], "oth_gate": [],
     "q_cue": [], "q_per": [], "q_oth": []}
for r in rows:
    text = tok.apply_chat_template([{"role": "user", "content": r["text"] + TAIL}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    q0 = text.find(r["text"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc.input_ids, enc.offset_mapping
    covering = lambda a, b: [i for i, (s, e) in enumerate(offs) if s < q0 + b and e > q0 + a]
    nums = {}                                    # written number -> its token positions
    for mm in re.finditer(r"(?<![\d.,])\d+(?![\d])", r["text"]):
        nums.setdefault(int(mm[0]), []).extend(covering(mm.start(), mm.end()))
    all_num = sorted({t for v in nums.values() for t in v})
    W, G, Q = run(ids)
    story = covering(0, len(r["text"]))
    sent_end = [mm.start() for mm in re.finditer(r"[.?!](?=\s|$)", r["text"])]
    start = 0
    per_tok = set()
    for e in sent_end:
        t = covering(e, e + 1)
        if not t:
            continue
        t = t[-1]
        per_tok.add(t)
        own = [i for i in all_num if start <= offs[i][0] - q0 < e]
        R["per_own"].append(W[t, own].sum() if own else np.nan)
        R["per_all"].append(W[t, all_num].sum() if all_num else np.nan)
        R["per_gate"].append(G[t]); R["q_per"].append(Q[t])
        start = e + 1
    for t in story:
        if t not in per_tok:
            R["oth_gate"].append(G[t]); R["q_oth"].append(Q[t])
    for c, a in stated(r):
        cue = tok(c, add_special_tokens=False).input_ids
        Wc, Gc, Qc = run(ids + cue)
        right = nums.get(a, [])
        other = [i for i in all_num if i not in right]
        R["cue_right"].append(Wc[-1, right].sum()); R["cue_other"].append(Wc[-1, other].sum() if other else np.nan)
        R["cue_gate"].append(Gc[-1]); R["q_cue"].append(Qc[-1])
    print(f"{r['id']} numbers {len(nums)} sentences {len(sent_end)}", flush=True)

np.savez(out.replace(".txt", ".npz"), **{k: np.array(v) for k, v in R.items() if k.startswith("q_")})
f = lambda v: f"median {np.nanmedian(v):.3f} (mean {np.nanmean(v):.3f}, n {np.sum(np.isfinite(v))})"
qc, qp, qo = (np.array(R[k]) for k in ("q_cue", "q_per", "q_oth"))
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
L = [f"head_look layer {LAYER} head {HEAD}, {len(rows)} problems ({src})",
     f"  cue (asked for a number): attention on the right number {f(R['cue_right'])}; on the other numbers {f(R['cue_other'])}",
     f"     gate {f(R['cue_gate'])}",
     f"  period (end of a story sentence, nobody asked): attention on that sentence's numbers {f(R['per_own'])}; on all numbers {f(R['per_all'])}",
     f"     gate {f(R['per_gate'])}",
     f"  other story words: gate {f(R['oth_gate'])}",
     f"  query direction: cos(mean cue, mean period) {cos(qc.mean(0), qp.mean(0)):.3f}; cos(mean cue, mean other) {cos(qc.mean(0), qo.mean(0)):.3f}; "
     f"cos(cue - other, period - other) {cos(qc.mean(0) - qo.mean(0), qp.mean(0) - qo.mean(0)):.3f}"]
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
