#!/usr/bin/env python
# Table from neuron-weights-v1.json, with a random matrix of the same shape as the reference.
import json, sys, torch
rows = json.load(open('neuron-weights-v1.json'))
torch.manual_seed(0)
w = torch.randn(7680, 2048)
G = (w.T @ w).double(); e = torch.linalg.eigvalsh(G).flip(0); c = torch.cumsum(e, 0) / e.sum()
r90, r99 = int((c < .9).sum()) + 1, int((c < .99).sum()) + 1
u = w / w.norm(dim=1, keepdim=True); C = (u @ u.T).abs(); C.fill_diagonal_(0); near = float(C.max(1).values.median())
print(f'Random matrix, same shape: directions for 90%/99% energy {r90}/{r99}, closest partner median {near:.3f}')
print('Quiet = share of total loudness held by the quietest 50% of neurons (even spread would be 0.50)')
print('layer quiet50 merge>.9 merge>.7 merge>.5 gate~level|>.5| closest(g/l/w)    rank99 w1/w3/w2  rank90 w1/w3/w2')
for r in rows:
    m = r['mergeable_share']; p = r['closest_partner_median']
    print(f"{r['layer']:5d} {r['quiet_share']['0.5']:7.3f} {m['0.9']:8.4f} {m['0.7']:8.4f} {m['0.5']:8.4f} {r['gate_vs_level_cos']['share_abs_over_0_5']:15.3f} "
          f"{p['gate']:.2f}/{p['level']:.2f}/{p['write']:.2f}  {r['rank_w1'][1]:5d}/{r['rank_w3'][1]}/{r['rank_w2'][1]}  {r['rank_w1'][0]:5d}/{r['rank_w3'][0]}/{r['rank_w2'][0]}")
print(len(rows), 'layers read')
