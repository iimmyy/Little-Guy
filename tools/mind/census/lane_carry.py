#!/usr/bin/env python3
"""Lane carry test (2026-11-03 night, after c5m2): can a note written at layer A reach layer B unchanged?
c5m2 pushed c5's wire 397 at layer 9 and the push came out at layer 17 bent and growing (the calibration diverged):
397 is quiet for layers 18-23 (c5 chose it for that) but layers 10-17 read it. A misfit alarm at layer 9 needs a lane
those in-between layers leave alone.
For each lane: add d * lane to the stream after layer A on his own thinking words, then at layer B measure
  carry:  how much of the push is still on the lane (1.0 = arrived as written)
  spill:  size of the change on every other direction, relative to d (0 = the layers in between ignored it)
  drift:  mean KL of the next-word odds vs no push, and how often the top word changes
Lanes: c5's wire, the lane the in-between layers' input weights read least (weights only, like c5's own pick but over
layers A+1..B, and optionally also quiet for the layers after B), and random lanes as the control.
v2 (same night): also the carry after EVERY layer from A to B (where does the note go?), so a sanity row (carry 1.0
right after A) comes for free.
v4: the weight census also counts WRITES (each layer's output weights times its post-norm per dimension): 397 is the
loudest-written, least-read dimension at every layer 9-17, a lane the layers keep at a set point. A note can only
survive on a lane that is quiet both ways ("quiet r+w"). v5 "private": also unread by the layers after B (the r+w
lane carried 0.94 but layers 18-23 read it 2.2x average: drift KL 8.4).
only=<lane name prefix>: test just those lanes (plus nothing else).
split=<layer>: also how much of that layer's change on the lane comes from its attention/conv write vs its MLP write
(v2 found layer 10 wipes 70-97% of a note on a quiet lane).
usage: lane_carry.py <model folder> <thinking.jsonl> <out.txt> [a=9 b=17 d=3.5 n=24 rand=4 split= dev=cuda]
"""
import json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL

hf, src, out = sys.argv[1:4]
opt = dict(a.split("=", 1) for a in sys.argv[4:])
A, B, D_, N, NR, DEV = (int(opt.get("a", 9)), int(opt.get("b", 17)), float(opt.get("d", 3.5)), int(opt.get("n", 24)),
                        int(opt.get("rand", 4)), opt.get("dev", "cuda"))
WIRE = json.load(open(os.path.join(hf, "pam.json")))["compressor"]["wire"] if os.path.exists(os.path.join(hf, "pam.json")) else 397
p = _lfm3(DEV, hf)
tok, m = p.tok, p.model
lay = m.model.layers
H = m.config.hidden_size


def read_matrix(lo, hi):
    """how strongly layers lo..hi read each direction of the stream (sum of their normalised input weights' Gram)"""
    Mt = torch.zeros(H, H)
    with torch.no_grad():
        for i in range(lo, hi + 1):
            l2 = lay[i]
            ins = [(l2.ffn_norm.weight, l2.feed_forward.w1.weight), (l2.ffn_norm.weight, l2.feed_forward.w3.weight)]
            if l2.is_attention_layer:
                ins += [(l2.operator_norm.weight, w.weight) for w in (l2.self_attn.q_proj, l2.self_attn.k_proj, l2.self_attn.v_proj, l2.self_attn.gate_proj)]
            else:
                ins += [(l2.operator_norm.weight, l2.conv.in_proj.weight)]
            for nw, W in ins:
                Rm = (W.float() * nw.float()[None, :]).cpu()
                Mt += Rm.T @ Rm / Rm.pow(2).sum()
    return Mt / Mt.trace()


def write_matrix(lo, hi):
    """how strongly layers lo..hi write each direction (output weights scaled by the post-norm, Gram, normalised)"""
    Mt = torch.zeros(H, H)
    with torch.no_grad():
        for i in range(lo, hi + 1):
            l2 = lay[i]
            outs = [(l2.post_ffn_norm.weight, l2.feed_forward.w2.weight),
                    (l2.post_operator_norm.weight, l2.self_attn.o_proj.weight if l2.is_attention_layer else l2.conv.out_proj.weight)]
            for nw, W in outs:
                Rm = (W.float() * nw.float()[:, None]).cpu()
                Mt += Rm @ Rm.T / Rm.pow(2).sum()
    return Mt / Mt.trace()


Mmid, Mlate = read_matrix(A + 1, B), read_matrix(B + 1, len(lay) - 1)
Wmid = write_matrix(A + 1, B)
lanes = {}
e = torch.zeros(H); e[WIRE] = 1
lanes[f"c5 wire {WIRE}"] = e
ev, V = torch.linalg.eigh(Mmid)
lanes[f"quiet for {A + 1}-{B}"] = V[:, 0]
ev2, V2 = torch.linalg.eigh(Mmid + Mlate)
lanes[f"quiet for {A + 1}-{len(lay) - 1}"] = V2[:, 0]
ev3, V3 = torch.linalg.eigh(Mmid + Wmid)
lanes[f"quiet r+w {A + 1}-{B}"] = V3[:, 0]
ev4, V4 = torch.linalg.eigh(Mmid + Mlate + Wmid)       # v5: nobody after A hears it, nobody up to B resets it
lanes[f"private {A + 1}-{len(lay) - 1}"] = V4[:, 0]
g = torch.Generator().manual_seed(0)
for r in range(NR):
    v = torch.randn(H, generator=g); lanes[f"random {r}"] = v / v.norm()
if opt.get("only"):
    lanes = {k: v for k, v in lanes.items() if any(k.startswith(o) for o in opt["only"].split(","))}
L = [f"lane carry: push {D_} after layer {A}, read after layer {B}; {N} thinking passages"]
for nm, v in lanes.items():
    L.append(f"  {nm:22s} weight-read by {A + 1}-{B}: {float(v @ Mmid @ v) * H:.3f} x average, by {B + 1}-{len(lay) - 1}: {float(v @ Mlate @ v) * H:.3f} x average, written by {A + 1}-{B}: {float(v @ Wmid @ v) * H:.3f} x average")

rows = [r for r in map(json.loads, open(src)) if r.get("thinking")][:N]
seqs = [tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True) + r["thinking"][:2500], add_special_tokens=False).input_ids[:800] for r in rows]
state = {"push": None, "hb": None, "hs": {}}


def add_push(mod, a, o):
    if state["push"] is None:
        return o
    if isinstance(o, tuple):
        return (o[0] + state["push"],) + tuple(o[1:])
    return o + state["push"]


lay[A].register_forward_hook(add_push)
for i in range(A, B + 1):          # registered after the push hook, so layer A's capture already has the push in it
    lay[i].register_forward_hook(lambda mod, a, o, i=i: state["hs"].__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].float()))


SPLIT = int(opt["split"]) if opt.get("split") else None
if SPLIT is not None:
    lay[SPLIT].post_operator_norm.register_forward_hook(lambda mod, a, o: state["hs"].__setitem__("op", o[0].float()))
    lay[SPLIT].post_ffn_norm.register_forward_hook(lambda mod, a, o: state["hs"].__setitem__("ffn", o[0].float()))


def run(ids, push):
    state["push"] = push
    with torch.no_grad():
        lg = m(input_ids=torch.tensor([ids], device=DEV)).logits[0].float()
    return {i: h.clone() for i, h in state["hs"].items()}, torch.log_softmax(lg, -1)


base = [run(ids, None) for ids in seqs]
for nm, v in lanes.items():
    vd = v.to(DEV)
    push = (D_ * vd).to(m.dtype)
    car, spl, kl, flip, by = [], [], [], [], {i: [] for i in range(A, B + 1)}
    parts = {"op": [], "ffn": []}
    for ids, (hs0, lp0) in zip(seqs, base):
        hs1, lp1 = run(ids, push)
        for i in by:
            by[i].append(((hs1[i] - hs0[i]) @ vd) / D_)
        for k in parts if SPLIT is not None else ():
            parts[k].append(((hs1[k] - hs0[k]) @ vd) / D_)
        dh = hs1[B] - hs0[B]
        c = dh @ vd
        car.append(c / D_)
        spl.append((dh - c[:, None] * vd[None]).norm(dim=-1) / D_)
        kl.append((lp0.exp() * (lp0 - lp1)).sum(-1))
        flip.append((lp0.argmax(-1) != lp1.argmax(-1)).float())
    car, spl, kl, flip = (torch.cat(x) for x in (car, spl, kl, flip))
    L.append(f"  {nm:22s} carry median {float(car.median()):+.3f} (10-90%: {float(car.quantile(.1)):+.3f} .. {float(car.quantile(.9)):+.3f}), "
             f"spill median {float(spl.median()):.3f}, drift KL {float(kl.mean()):.4f}, top word changed {float(flip.mean()):.1%}")
    L.append(f"    carry after each layer: " + "  ".join(f"{i}: {float(torch.cat(by[i]).median()):+.2f}" for i in by))
    if SPLIT is not None:
        L.append(f"    layer {SPLIT}'s change on the lane: attention/conv write {float(torch.cat(parts['op']).median()):+.2f}, "
                 f"MLP write {float(torch.cat(parts['ffn']).median()):+.2f} (x the push)")
    print("\n".join(L[-3:]), flush=True)
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
