#!/usr/bin/env python3
"""Goal: are there angles of the page his word choice cannot see, and do his neurons see them? Weights only, no model run.
Power of a direction = how strongly a reader responds to the page along it (eigenvalues of R^T R, R = reader rows with its norm weights)."""
import os, sys
import numpy as np, torch
from safetensors import safe_open
p = os.path.expanduser('~/.littleguy/hf/lfm3-1.7b/model.safetensors')
if not os.path.exists(p):
    print('no weights on this Mac at', p); sys.exit()
f = safe_open(p, 'pt')
keys = set(f.keys())
G = lambda n: f.get_tensor(n).float()
U = G('model.embed_tokens.weight')
c = G('model.embedding_norm.weight')
print('word vectors', tuple(U.shape), 'final norm weights: min %.3f median %.3f max %.3f' % (c.min(), c.median(), c.max()))
def spec(R):
    e = torch.linalg.eigvalsh(R.T @ R).clamp(min=0).flip(0)
    s = e.cumsum(0) / e.sum()
    n = lambda q: int((s < q).sum()) + 1
    return e, f"directions holding 50% of the power {n(.5)}, 90% {n(.9)}, 99% {n(.99)}, 99.9% {n(.999)} of {len(e)}; weakest vs strongest response {float((e[-1]/e[0]).sqrt()):.2e}; directions under 1% of the median response {int(((e/e.median()).sqrt()<.01).sum())}"
e, t = spec(U * c)
print('word head:', t)
Q = torch.linalg.eigh((U * c).T @ (U * c)).eigenvectors.flip(1)
blind = Q[:, -256:]
for L in (2, 8, 14, 20, 23):
    b = f'model.layers.{L}.'
    ci = G(b + 'ffn_norm.weight')
    w1, w3 = G(b + 'feed_forward.w1.weight') * ci, G(b + 'feed_forward.w3.weight') * ci
    _, t1 = spec(w1); _, t3 = spec(w3)
    R = torch.cat([w1, w3])
    tot = (R ** 2).sum()
    seen = ((R @ blind) ** 2).sum() / tot
    print(f'L{L} neuron gate reads: {t1}')
    print(f'L{L} neuron level reads: {t3}')
    print(f'L{L}: share of neuron read power on the 256 directions the word head sees least: {float(seen)*100:.1f}% (even share would be {256/2048*100:.1f}%)')
