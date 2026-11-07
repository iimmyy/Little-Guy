#!/usr/bin/env python
# Weights-only look at LFM3's neuron blocks. No model run, no pages.
# neuron i: output += w2[:,i] * silu(w1[i]·x) * (w3[i]·x)
# Input side is scaled by ffn_norm, output side by post_ffn_norm, since those are in the real path.
# Asks: which neurons are near silent, which are duplicates a merge could remove with no change,
# and how many directions each matrix really uses.
import json, struct, sys, time
import numpy as np, torch
P = sys.argv[1]; OUT = sys.argv[2]
f = open(P, 'rb'); L = struct.unpack('<Q', f.read(8))[0]; H = json.loads(f.read(L)); base = 8 + L
def get(k):
    a, b = H[k]['data_offsets']; f.seek(base + a); raw = f.read(b - a)
    assert H[k]['dtype'] == 'BF16'
    u = np.frombuffer(raw, dtype='<u2').astype('<u4') << 16      # BF16 is the top half of F32
    return torch.from_numpy(u.view('<f4').reshape(H[k]['shape']).copy())
def rank_at(W, fr):
    G = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
    e = torch.linalg.eigvalsh(G.double()).flip(0).clamp(min=0)
    c = torch.cumsum(e, 0) / e.sum()
    return [int((c < x).sum()) + 1 for x in fr]
FR = [0.9, 0.99]; TH = [0.95, 0.9, 0.8, 0.7, 0.5]
rows = []; t0 = time.time()
for n in range(24):
    p = f'model.layers.{n}.'
    w1 = get(p + 'feed_forward.w1.weight'); w3 = get(p + 'feed_forward.w3.weight'); w2 = get(p + 'feed_forward.w2.weight')
    a = get(p + 'ffn_norm.weight'); b = get(p + 'post_ffn_norm.weight')
    w1 = w1 * a; w3 = w3 * a; w2 = (w2.T * b).T
    n1 = w1.norm(dim=1); n3 = w3.norm(dim=1); n2 = w2.norm(dim=0)
    loud = n1 * n3 * n2; s, _ = torch.sort(loud); cum = torch.cumsum(s, 0) / s.sum()
    N = len(s)
    r = dict(layer=n,
      quiet_share={str(q): float(cum[int(N*q)-1]) for q in (0.1, 0.3, 0.5, 0.7)},
      loud_ratio_p99_p1=float(s[int(N*.99)] / s[int(N*.01)].clamp(min=1e-12)))
    u1 = w1 / n1[:, None].clamp(min=1e-12); u3 = w3 / n3[:, None].clamp(min=1e-12); u2 = (w2 / n2.clamp(min=1e-12)).T
    gl = (u1 * u3).sum(1)
    r['gate_vs_level_cos'] = dict(mean=float(gl.mean()), p10=float(gl.quantile(.1)), p90=float(gl.quantile(.9)),
        share_abs_over_0_5=float((gl.abs() > .5).float().mean()), share_abs_over_0_8=float((gl.abs() > .8).float().mean()))
    C1 = u1 @ u1.T; C3 = (u3 @ u3.T).abs(); C2 = (u2 @ u2.T).abs()
    for C in (C1, C3, C2): C.fill_diagonal_(0)
    r['closest_partner_median'] = dict(gate=float(C1.max(1).values.median()), level=float(C3.max(1).values.median()), write=float(C2.max(1).values.median()))
    # a neuron can merge exactly when another has the same gate AND (same level OR same write)
    M = torch.minimum(C1, torch.maximum(C3, C2)).max(1).values
    r['mergeable_share'] = {str(t): float((M > t).float().mean()) for t in TH}
    del C1, C2, C3
    r['rank_w1'] = rank_at(w1, FR); r['rank_w3'] = rank_at(w3, FR); r['rank_w2'] = rank_at(w2, FR)
    rows.append(r)
    print(n, 'quiet50 %.3f' % r['quiet_share']['0.5'], 'merge@.7 %.4f' % r['mergeable_share']['0.7'],
          'gate~level|>.5| %.3f' % r['gate_vs_level_cos']['share_abs_over_0_5'],
          'rank w1/w3/w2 99%%', r['rank_w1'][1], r['rank_w3'][1], r['rank_w2'][1], '%.0fs' % (time.time() - t0), flush=True)
    json.dump(rows, open(OUT, 'w'), indent=1)
print('LOOK_DONE', flush=True)
