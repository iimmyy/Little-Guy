#!/usr/bin/env python3
"""Where in LFM3 should E4B's rooms go? The plug has to fit the socket: the graft's entry translator turns LFM3's
stream into what E4B's room 8 (or 10) expects to read, so the best site is the LFM3 layer whose stream a straight-line
map turns into that most faithfully (model stitching: Lenc & Vedaldi 2015, Bansal et al. 2021).

Both models read the same WikiText; their streams are paired at the spots where both word lists end a piece at the
same letter. For every LFM3 layer a ridge map is fitted on the training text and scored on held-out text:
  r2      share of E4B's stream variance explained (dominated by E4B's giant dimension)
  r2_dim  the same per dimension, averaged (every dimension counts the same)
  cos     how closely each predicted vector points the same way as the real one (what the room's norm keeps)

Each room's best map is saved next to out.json as site_map_room<r>_lfm<k>.pt (the graft's entry translator start).

usage: stitch_site.py <wikitext folder> <out.json> [rooms=8,10 fit=24000 test=4000 dev=cuda]
"""
import json, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../census"))
from patients import load

data, out = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
ROOMS = [int(r) for r in opt.get("rooms", "8,10").split(",")]
NFIT, NTEST, DEV = int(opt.get("fit", 24000)), int(opt.get("test", 4000)), opt.get("dev", "cuda")
WIN = 1024                                                     # characters' worth is set by LFM3 pieces per window


def text(split, n_chars):
    return open(f"{data}/wt103-{split}.txt").read()[:n_chars]          # parquet text joined, blank lines dropped


def windows(t, tok, n_pos):
    """cut the text into windows of WIN LFM3 pieces' worth of characters, enough to reach n_pos paired spots"""
    enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
    off, ws, i = enc.offset_mapping, [], 0
    while i + WIN < len(off) and len(ws) * WIN * 0.8 < n_pos:
        ws.append(t[off[i][0]:off[i + WIN - 1][1]])
        i += WIN
    return ws


def stream(p, s, layers):
    """the stream after each wanted layer, and each piece's end letter; start-of-text in front, not counted"""
    enc = p.tok(s, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([[p.tok.bos_token_id] + enc.input_ids], device=DEV)
    hs = p.model.model(input_ids=ids, output_hidden_states=True).hidden_states
    return {k: hs[k][0, 1:].float().cpu() for k in layers}, [e for _, e in enc.offset_mapping]


def collect(split, n_chars, n_pos):
    lfm = load("lfm3", DEV)
    ws = windows(text(split, n_chars), lfm.tok, n_pos)
    L = len(lfm.model.model.layers)
    X, ends_b = [], []
    with torch.no_grad():
        for s in ws:
            h, e = stream(lfm, s, range(1, L + 1))                # hs[k] = after LFM3 layer k-1
            X.append(h); ends_b.append(e)
    del lfm; torch.cuda.empty_cache()
    e4b = load("e4b", DEV)
    Y, keep = [], []
    with torch.no_grad():
        for s, eb in zip(ws, ends_b):
            h, ea = stream(e4b, s, ROOMS)                         # hs[r] = input of room r
            pos_a = {e: j for j, e in enumerate(ea)}
            pairs = [(i, pos_a[e]) for i, e in enumerate(eb) if e in pos_a]
            keep.append(torch.tensor([i for i, _ in pairs]))
            Y.append({r: h[r][[j for _, j in pairs]] for r in ROOMS})
    del e4b; torch.cuda.empty_cache()
    Xs = {k: torch.cat([x[k][kp] for x, kp in zip(X, keep)])[:n_pos] for k in X[0]}
    Ys = {r: torch.cat([y[r] for y in Y])[:n_pos] for r in ROOMS}
    print(f"  {split}: {len(ws)} windows, {sum(map(len, keep))} paired spots of {sum(map(len, ends_b))} LFM3 pieces",
          flush=True)
    return Xs, Ys


Xf, Yf = collect("train-00000-of-00002", 4_000_000, NFIT)
Xt, Yt = collect("validation-00000-of-00001", 600_000, NTEST)
res, best_r2 = {}, {r: (-1e9, None) for r in ROOMS}
for r in ROOMS:
    y, yt = Yf[r].double(), Yt[r].double()
    my = y.mean(0)
    cos0 = torch.nn.functional.cosine_similarity(my.expand_as(yt), yt, dim=-1).mean()
    print(f"  E4B room {r:2}: guessing its average stream alone already gives cos {cos0:.3f}", flush=True)
    res[f"room{r}_mean_only_cos"] = float(cos0)
    for k in sorted(Xf):
        x, xt = Xf[k].double(), Xt[k].double()
        mx = x.mean(0)
        xc, yc = x - mx, y - my
        cut = int(0.85 * len(x))                                  # ridge strength picked on the fit text's last 15%
        e, V = torch.linalg.eigh(xc[:cut].T @ xc[:cut])
        P = V.T @ (xc[:cut].T @ yc[:cut])
        scale, best = e.mean(), None
        for f in (1e-3, 1e-2, 1e-1, 1, 10):
            Wf = V @ (P / (e + f * scale)[:, None])
            sse = (xc[cut:] @ Wf - yc[cut:]).pow(2).sum()
            best = (sse, f) if best is None or sse < best[0] else best
        lam = best[1] * scale
        W = torch.linalg.solve(xc.T @ xc + lam * torch.eye(x.shape[1], dtype=x.dtype), xc.T @ yc)
        pred = (xt - mx) @ W + my
        err, var = (pred - yt).pow(2), (yt - yt.mean(0)).pow(2)
        r2 = 1 - err.sum() / var.sum()
        r2d = (1 - err.sum(0) / var.sum(0)).mean()
        cos = torch.nn.functional.cosine_similarity(pred, yt, dim=-1).mean()
        if r2 > best_r2[r][0]:
            best_r2[r] = (float(r2), k - 1)
            torch.save({"W": W.float(), "mx": mx.float(), "my": my.float(), "r2": float(r2)},
                       os.path.join(os.path.dirname(out), f"site_map_room{r}_lfm{k - 1}.pt"))
        res[f"room{r}_lfm{k - 1}"] = {"room": r, "lfm_layer": k - 1, "ridge": best[1], "r2": float(r2), "r2_dim": float(r2d),
                                      "cos": float(cos)}
        print(f"  E4B room {r:2} input <- LFM3 after layer {k - 1:2}: r2 {r2:.3f}  r2_dim {r2d:.3f}  cos {cos:.3f}  ridge {best[1]:g}",
              flush=True)
json.dump(res, open(out, "w"), indent=1)
print("SITE_DONE")
