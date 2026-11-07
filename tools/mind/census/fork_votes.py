#!/usr/bin/env python3
"""How does he know he is done? Count the votes (Immy 2026-10-29: "what causes the sentence, how does he know he's
done? think of math"). His next word is picked by score(word) = final stream . that word's output row, and the final
stream is a plain sum: the word itself plus what every block (each layer's attention/conv and its MLP, after their
sandwich norms) wrote. So at the fork, the word right before the next sentence after "Thus answer is X." (X right),
    vote(block) = block's write . g_final * (row_handoff - row_doubt) / rms(final stream)
adds up exactly to the score gap between handing in ("I'll produce final output", "We need to end with a line") and
doubting ("But let's double-check"). Per moment, per block; then per MLP neuron inside the blocks that matter:
    vote(neuron j) = a_j * (g_ffn * w2[:, j]) / rms(MLP output) . g_final * (row_h - row_d) / rms(final)
  blocks    which blocks vote differently at hand-in moments vs doubt moments (train, and the same on test)
  neurons   the neurons with the biggest vote gap (train), with test
  check     the votes add up to the real score gap (should match to rounding)

usage: fork_votes.py <patient> <train answers> <test answers> <out folder> [dev=mps cut=7000 top=40]
"""
import json, os, re, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right
from patients import load

name, src_a, src_b, out = ARGS[:4]
opt = dict(a.split("=", 1) for a in ARGS[4:])
DEV, CUT, TOP = opt.get("dev", "mps"), int(opt.get("cut", 7000)), int(opt.get("top", 40))
os.makedirs(out, exist_ok=True)
TAIL = "\n\nEnd with a line: Answer: <your answer>"
OPEN = "\n"
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:)\s*\$?([^\n$]{1,40}?)\$?(?=[.\n]|$)", re.I)
DOUBT = re.compile(r"(But|Wait|However|Hmm|Let me|Let's|Actually|Alternatively|Double|Check|Hold on)\b", re.I)
HANDOFF = re.compile(r"(I'll (produce|write|output|give|present|answer)|We need to (end|output|produce|write|present)"
                     r"|Now (we need to )?(output|produce|write)|So (the )?final (output|line)|Thus,? final\.?\s*$"
                     r"|Thus,? I'll|End with|So final answer|Thus final answer)", re.I)

p = load(name, DEV)
tok, m_ = p.tok, p.model
lay = m_.model.layers
L = len(lay)
g_final = m_.model.embedding_norm.weight.float()
eps_f = m_.model.embedding_norm.variance_epsilon if hasattr(m_.model.embedding_norm, "variance_epsilon") else 1e-5
W_U = m_.lm_head.weight
cap = {}


def hook(key):
    return lambda mod, a, o: cap.__setitem__(key, o[0])


m_.model.embed_tokens.register_forward_hook(hook(("emb",)))
for i, l in enumerate(lay):
    l.post_operator_norm.register_forward_hook(hook(("op", i)))
    l.post_ffn_norm.register_forward_hook(hook(("ffn", i)))
    l.feed_forward.w2.register_forward_pre_hook(lambda mod, a, i=i: cap.__setitem__(("act", i), a[0][0]))
    l.feed_forward.w2.register_forward_hook(hook(("pre", i)))       # MLP output before its sandwich norm
BLOCKS = [("emb",)] + [(k, i) for i in range(L) for k in ("op", "ffn")]


def forks(r, think):
    ms = []
    for m in CLAIM.finditer(think):
        x = next(g for g in m.groups() if g is not None).strip().rstrip(".").strip()
        if not x or not right({**r, "reply": x}, x):
            continue
        e = m.end()
        while e < len(think) and not (think[e] == "\n" or think[e] == "." and (e + 1 == len(think) or think[e + 1] in " \n")):
            e += 1
        s = e + 1
        while s < len(think) and think[s] in " \n":
            s += 1
        if s >= len(think) - 2:
            continue
        nxt = think[s:s + 60]
        kind = "handoff" if HANDOFF.match(nxt) else "doubt" if DOUBT.match(nxt) else None
        if kind:
            ms.append((kind, s))                           # s = first character of the next sentence
    return ms


def collect(path):
    rows = [r for r in map(json.loads, open(path)) if r["thinking"]]
    V, NV, kind, nxt_id, gap_real = [], {i: [] for i in range(L)}, [], [], []
    keep_rows = []
    with torch.no_grad():
        for n, r in enumerate(rows):
            think = OPEN + r["thinking"]
            ms = forks(r, think)
            if not ms:
                continue
            pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True),
                     add_special_tokens=False).input_ids
            enc = tok(think, add_special_tokens=False, return_offsets_mapping=True)
            ids, starts = enc.input_ids[:CUT], np.array([s for s, _ in enc.offset_mapping[:CUT]])
            pts = []
            for k_, s in ms:
                t = int(np.searchsorted(starts, s, side="right")) - 1   # token holding the next sentence's first char
                if 1 <= t < len(ids):
                    pts.append((k_, t - 1, ids[t]))                     # decide at t-1, the word he chose is ids[t]
            if not pts:
                continue
            cap.clear()
            m_.model(input_ids=torch.tensor([pr + ids], device=DEV))
            at = torch.tensor([len(pr) + t for _, t, _ in pts], device=DEV)
            W = torch.stack([cap[b][at].float() for b in BLOCKS])          # blocks x moments x hidden
            V.append(W.cpu()); kind += [k_ for k_, _, _ in pts]; nxt_id += [w for _, _, w in pts]
            keep_rows += [n] * len(pts)
            for i in range(L):
                a = cap[("act", i)][at].float()
                rms = cap[("pre", i)][at].float().pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
                NV[i].append((a / rms).half().cpu())                        # activation already divided by the norm's scale
            if (n + 1) % 20 == 0:
                print(f"  {os.path.basename(path)} {n + 1}/{len(rows)}: handoff {kind.count('handoff')} doubt {kind.count('doubt')}", flush=True)
    return torch.cat(V, 1), {i: torch.cat(v).float() for i, v in NV.items()}, np.array(kind), np.array(nxt_id), np.array(keep_rows)


Va, NVa, ka, ida, ra = collect(src_a)
Vb, NVb, kb, idb, rb = collect(src_b)
ha, hb = ka == "handoff", kb == "handoff"
with torch.no_grad():
    rh = W_U[torch.tensor(ida[ha], device=DEV)].float().mean(0); rd = W_U[torch.tensor(ida[~ha], device=DEV)].float().mean(0)
    d = (g_final * (rh - rd)).cpu()                                      # the hand-in-minus-doubt listener (train words)


def votes(V):
    S = V.sum(0)                                                         # final stream before its norm
    rmsf = S.pow(2).mean(-1, keepdim=True).add(eps_f).sqrt()
    return (V @ d) / rmsf[None, :, 0], (S @ d) / rmsf[:, 0]


torch.save({"Va": Va.half(), "Vb": Vb.half(), "NVa": NVa, "NVb": NVb, "ka": ka, "kb": kb, "ida": ida, "idb": idb,
            "ra": ra, "rb": rb}, os.path.join(out, f"fork_votes_{name}_raw.pt"))     # kept before any later step can fail
va, ga = votes(Va); vb, gb = votes(Vb)
with torch.no_grad():
    real = []
    for V_ in (Vb,):
        S = V_.sum(0).to(DEV, W_U.dtype)
        lg = m_.lm_head(m_.model.embedding_norm(S)).float()
        hi, di = torch.tensor(ida[ha], device=DEV), torch.tensor(ida[~ha], device=DEV)
        real = (lg[:, hi].mean(1) - lg[:, di].mean(1)).cpu()            # mean of rows = mean of their scores
lines = [f"how {name} decides it is done: votes at the fork after a right answer (next sentence: hand-in vs doubt)",
         f"train: hand-in {ha.sum()} doubt {(~ha).sum()}; test: hand-in {hb.sum()} doubt {(~hb).sum()}",
         f"words he chose at hand-in (train, top): {[tok.decode([w]) for w in np.unique(ida[ha])[:12]]}",
         f"words he chose at doubt (train, top): {[tok.decode([w]) for w in np.unique(ida[~ha])[:12]]}",
         f"total score gap hand-in minus doubt (test, median): at hand-in {float(gb[hb].median()):+.2f}, at doubt {float(gb[~hb].median()):+.2f}",
         f"check: votes sum vs the model's own scores, correlation {np.corrcoef(gb.numpy(), real.numpy())[0, 1]:.3f}",
         "block        vote at hand-in / at doubt (train medians)   gap train / test"]
gap_tr = va[:, ha].median(1).values - va[:, ~ha].median(1).values
gap_te = vb[:, hb].median(1).values - vb[:, ~hb].median(1).values
order = torch.argsort(-gap_tr.abs())
for b in order[:20].tolist():
    nm = "/".join(map(str, BLOCKS[b]))
    lines.append(f"  {nm:10s}  {float(va[b, ha].median()):+7.2f} / {float(va[b, ~ha].median()):+7.2f}"
                 f"                       {float(gap_tr[b]):+6.2f} / {float(gap_te[b]):+6.2f}")
lines.append("neurons (MLP), vote gap hand-in minus doubt, train / test, and activation hand-in / doubt (train mean):")
cand = []
for i in range(L):
    l = lay[i]
    col = (l.post_ffn_norm.weight.float()[:, None] * l.feed_forward.w2.weight.float()).cpu()   # hidden x inter
    per = (d @ col)                                                                            # vote per unit activation
    rmsf_a = Va.sum(0).pow(2).mean(-1).add(eps_f).sqrt(); rmsf_b = Vb.sum(0).pow(2).mean(-1).add(eps_f).sqrt()
    na = NVa[i] * per / rmsf_a[:, None]; nb = NVb[i] * per / rmsf_b[:, None]
    gtr = na[ha].mean(0) - na[~ha].mean(0); gte = nb[hb].mean(0) - nb[~hb].mean(0)
    for j in torch.argsort(-gtr.abs())[:TOP].tolist():
        cand.append((float(gtr[j]), float(gte[j]), i, j, float(NVa[i][ha, j].mean()), float(NVa[i][~ha, j].mean()), float(per[j])))
cand.sort(key=lambda c: -abs(c[0]))
for g1, g2, i, j, sa, sd_, pj in cand[:TOP]:
    lines.append(f"  layer {i:2d} neuron {j:5d}   {g1:+.3f} / {g2:+.3f}    {sa:+.3f} / {sd_:+.3f}   (vote per unit {pj:+.3f})")
print("\n".join(lines))
open(os.path.join(out, f"fork_votes_{name}.txt"), "w").write("\n".join(lines) + "\n")
torch.save({"cand": cand, "d": d, "blocks": BLOCKS, "va": va, "vb": vb, "ka": ka, "kb": kb, "ra": ra, "rb": rb},
           os.path.join(out, f"fork_votes_{name}.pt"))
print("FORK_VOTES_DONE")
