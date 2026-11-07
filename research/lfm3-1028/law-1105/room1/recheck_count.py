#!/usr/bin/env python3
"""Sizes both sides of knowing right and wrong. Rule fixed in GOAL-recheck-v1.md. No model."""
import collections, json, random, re
import numpy as np
B = '../../science/final-lfm3-1104/lfm3-base-%s.jsonl'
DOUBT = re.compile(r"\b(wait|actually|hold on|hmm|oops|double.?check|re-?check|verify|recalc\w*|recompute|re-compute|let me check|let's check|let me verify|let's verify|not right|seems off|that's off|seems wrong)\b", re.I)
NUM = re.compile(r'(?<![A-Za-z_\d.])\d[\d,]*\.?\d*')
def digits(s):
    return s.replace(',', '').replace('.', '').lstrip('0')
def plain(s):
    return s.replace(',', '').rstrip('.')
T = collections.Counter()
came, gone = [], []
lengths, per_run = [], []
for run in ('g', 's1', 's2', 's3'):
    for l in open(B % run):
        r = json.loads(l)
        if r.get('src') == 'cfe' or not r.get('thinking'):
            continue
        th = r['thinking']
        T['runs'] += 1
        T['words'] += len(th.split())
        nums = [(m.start(), m.end(), plain(m.group())) for m in NUM.finditer(th)]
        spans = []
        last_end = -1
        for d in DOUBT.finditer(th):
            T['doubt'] += 1
            p = d.start()
            if p < last_end:
                T['inside_loop'] += 1
            anchor = None
            for s, e, v in reversed(nums):
                if e > p:
                    continue
                if p - e > 60:
                    break
                if len(digits(v)) >= 3:
                    anchor = (s, e, v)
                break
            if anchor is None:
                T['no_anchor'] += 1
                continue
            T['anchored'] += 1
            s0, e0, v0 = anchor
            between = 0
            hit = None
            for s, e, v in nums:
                if s <= d.end():
                    continue
                if s > d.end() + 2000:
                    break
                if v == v0 and between >= 1:
                    hit = e
                    break
                if v != v0:
                    between += 1
            item = dict(run=run, id=r['id'], at=p, v=v0, text=th[max(0, s0 - 160):d.end() + 520].replace('\n', ' / '))
            if hit is not None:
                T['came'] += 1
                spans.append((p, hit))
                lengths.append(len(th[p:hit].split()))
                last_end = max(last_end, hit)
                came.append(item)
            else:
                T['gone'] += 1
                gone.append(item)
        spans.sort()
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        w = sum(len(th[a:b].split()) for a, b in merged)
        T['loop_words'] += w
        per_run.append(w / max(len(th.split()), 1))
print(f"runs {T['runs']}, words {T['words']:,}")
print(f"doubt words {T['doubt']:,}; per 1,000 words {T['doubt']/T['words']*1000:.1f}")
print(f"no number anchor {T['no_anchor']:,} ({T['no_anchor']/T['doubt']*100:.1f}%); number rechecks {T['anchored']:,} ({T['anchored']/T['doubt']*100:.1f}%)")
print(f"CAME BACK {T['came']:,} ({T['came']/T['anchored']*100:.1f}% of number rechecks); DID NOT {T['gone']:,}")
print(f"loop words merged {T['loop_words']:,} = {T['loop_words']/T['words']*100:.1f}% of all thinking; median run {np.median(per_run)*100:.1f}%; median loop {int(np.median(lengths))} words, 90th percentile {int(np.percentile(lengths,90))}")
rng = random.Random(1107)
rng.shuffle(came)
rng.shuffle(gone)
for name, pile in (('came', came), ('gone', gone)):
    with open(f'recheck-{name}-sample-v1.txt', 'w') as h:
        for i, x in enumerate(pile[:40]):
            h.write(f"\n### {i} {x['run']} {x['id']} number {x['v']}\n{x['text']}\n")
