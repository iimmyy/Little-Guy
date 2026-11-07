#!/usr/bin/env python3
"""Description only, after GOAL-broken-v1 failed. Nothing here changes the verdict or fits anything."""
import sys
sys.argv = ['x', 'lfm3-broken-v1']
src = open('broken_score.py').read()
exec(src[:src.index('B, H = split')])
B, H = split == 'build', split == 'held'
print('--- surprise at the number itself, held-out')
sb, sr, sc = surp['broken'][H], surp['reformat'][H], surp['clean'][H]
print(f"median bits clean {np.nanmedian(sc):.1f} reformat {np.nanmedian(sr):.1f} broken {np.nanmedian(sb):.1f}")
print(f"same step broken > reformat {same(sb, sr):.3f}; pooled broken vs clean+reformat {auc(sb, np.concatenate([sc, sr])):.3f}; reformat > clean {same(sr, sc):.3f}")
print('--- does the read just follow surprise? rank correlation of (broken minus reformat) with surprise difference, held-out')
def spearman(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])
W = {'total@ffn23 (-)': -win['total'][1][..., win['total'][0].index('total@ffn23')],
     'held@ffn23 (-)': -win['held'][1][..., win['held'][0].index('held@ffn23')],
     'pair op9~ffn11 (-)': -win['pair'][1][..., win['pair'][0].index('pair:op9~ffn11')],
     'closed@L13 (-)': -win['closed'][1][..., win['closed'][0].index('closed@L13')],
     'next_surprise (+)': win['first'][1][..., win['first'][0].index('next_surprise')]}
for n, a in W.items():
    d = a[H, 1] - a[H, 2]
    print(f"{n:22s} corr with surprise gap {spearman(d, sb - sr):+.3f}; broken>reformat {same(a[H,1],a[H,2]):.3f}, broken>clean {same(a[H,1],a[H,0]):.3f}, reformat>clean {same(a[H,2],a[H,0]):.3f}")
print('--- surprise-matched, held-out, gap within 5 bits, then reformat MORE surprising than broken')
for lab, mask in (('within 5 bits', H & (np.abs(surp['broken'] - surp['reformat']) <= 5)), ('reformat more surprising', H & (surp['reformat'] > surp['broken']))):
    print(lab, 'n', int(mask.sum()), ' '.join(f"{n.split(' ')[0]}:{same(a[mask,1],a[mask,2]):.2f}" for n, a in W.items()))
print('--- conflict direction: broken above clean, held-out, for conflict as predicted')
tot = win['total'][1]; names = win['total'][0]
for nm in ('total@op10','total@op12','total@ffn13','total@op22','total@ffn23'):
    c = names.index(nm); print(nm, f"conflict higher after a wrong digit on {same(tot[H,1,c],tot[H,0,c]):.3f} of steps; reformat {same(tot[H,2,c],tot[H,0,c]):.3f}")
print('--- closed energy by layer, sign fixed to falling, held-out: L, broken>reformat, broken>clean, reformat>clean')
ec = -win['closed'][1]
for L in range(24):
    print(f"L{L:2d} {same(ec[H,1,L],ec[H,2,L]):.3f} {same(ec[H,1,L],ec[H,0,L]):.3f} {same(ec[H,2,L],ec[H,0,L]):.3f}")
print('--- where the best pairs sit: layer of each of the top 40 pairs on build')
pn, pa = win['pair']
sc2 = np.array([same(pa[B,1,c],pa[B,2,c]) for c in range(pa.shape[2])])
import re, collections as C
cnt = C.Counter()
for c in np.argsort(-np.abs(sc2-.5))[:40]:
    for x in re.findall(r'(?:op|ffn)(\d+)', pn[c]): cnt[int(x)] += 1
print(sorted(cnt.items()))
print('--- held read by half-layer, sign fixed so broken rises, held-out, top 8 on build')
hn, ha = win['held']
s3 = np.array([same(ha[B,1,c],ha[B,2,c]) for c in range(ha.shape[2])])
for c in np.argsort(-np.abs(s3-.5))[:8]:
    sg = 1 if s3[c] >= .5 else -1
    print(f"{hn[c]:14s} sign {sg:+d} build {s3[c]:.3f} held b>r {same(sg*ha[H,1,c],sg*ha[H,2,c]):.3f} b>c {same(sg*ha[H,1,c],sg*ha[H,0,c]):.3f} r>c {same(sg*ha[H,2,c],sg*ha[H,0,c]):.3f}")
print('--- is the mid-depth hint just next-word surprise? held-out')
ns = W['next_surprise (+)']
for n in ('closed@L13 (-)', 'pair op9~ffn11 (-)', 'held@ffn23 (-)'):
    a = W[n]
    print(f"{n:22s} corr of (broken-reformat) with next-word surprise gap {spearman(a[H,1]-a[H,2], ns[H,1]-ns[H,2]):+.3f}; (broken-clean) {spearman(a[H,1]-a[H,0], ns[H,1]-ns[H,0]):+.3f}")
for lab, mask in (('reformat next-word surprise at least as high as broken', H & (ns[:,2] >= ns[:,1])), ('both surprise gaps reversed', H & (ns[:,2] >= ns[:,1]) & (surp['reformat'] > surp['broken']))):
    print(lab, 'n', int(mask.sum()), ' '.join(f"{n.split(' ')[0]}:{same(a[mask,1],a[mask,2]):.2f}" for n, a in W.items()))
print('--- same reads on BUILD, for comparison')
for n, a in W.items():
    print(f"{n:22s} broken>reformat {same(a[B,1],a[B,2]):.3f}, broken>clean {same(a[B,1],a[B,0]):.3f}, reformat>clean {same(a[B,2],a[B,0]):.3f}")
mask = B & (surp['reformat'] > surp['broken'])
print('build, reformat more surprising, n', int(mask.sum()), ' '.join(f"{n.split(' ')[0]}:{same(a[mask,1],a[mask,2]):.2f}" for n, a in W.items()))
print('--- which reformat kinds: same step broken>reformat, all steps')
J = {}
for l in open('broken-steps-v2.jsonl'):
    j = json.loads(l); J[j['step']] = j
kind = np.array(['comma' if ',' in J[s]['clean'] else 'decimal+0' if '.' in J[s]['clean'] else 'int+.0' for s in steps])
for k in ('comma', 'decimal+0', 'int+.0'):
    mk = H & (kind == k)
    print(k, 'n', int(mk.sum()), f"surprise broken {np.nanmedian(surp['broken'][mk]):.1f} reformat {np.nanmedian(surp['reformat'][mk]):.1f}", ' '.join(f"{n.split(' ')[0]}:{same(a[mk,1],a[mk,2]):.2f}" for n, a in W.items()))
