import json, collections, numpy as np
d = np.load('gain1/plain-x.npz'); F = d['F']; names = list(d['names'])
rows = [json.loads(l) for l in open('gain1/plain-rows.jsonl')]
s = F[:, :, names.index('state_rms')]
ids = np.array([r['id'] for r in rows])
print('closes', s.shape[0], 'layers', s.shape[1], 'names', names)
print(' L  median r | gain p5-p95 spread % | CV % | within-problem CV %')
sp = []
for L in range(s.shape[1]):
    r = s[:, L]; g = 1 / r
    p5, p95 = np.percentile(g, [5, 95]); spread = 100 * (p95 / p5 - 1); sp.append(spread)
    within = []
    for i in set(ids):
        v = g[ids == i]
        if len(v) >= 2: within.append(v.std() / v.mean())
    print(f'{L:2d}  {np.median(r):9.3f} | {spread:6.1f} | {100*g.std()/g.mean():5.1f} | {100*np.median(within):5.1f}')
print('layers under 2%:', sum(x < 2 for x in sp), 'of', len(sp), '-> bar', 'DEAD' if sum(x < 2 for x in sp) > len(sp) / 2 else 'ALIVE')
R = [json.loads(l) for l in open('energy1/lfm3-plain-v1.jsonl')]
by = collections.defaultdict(list)
for x in R: by[x['L']].append(x['rr'])
print('neuron half: state growth rr p5/median/p95 by layer (120 states, close and random words)')
print(' '.join(f"L{L}:{np.percentile(v,5):.2f}/{np.median(v):.2f}/{np.percentile(v,95):.2f}" for L, v in sorted(by.items())))
