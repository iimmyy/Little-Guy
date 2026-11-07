#!/usr/bin/env python3
"""Goal: if a word is raised when he was unsure of it, how many words are raised? Saved reads only, no model.
m_self = top score minus runner-up score in nats; 2.94 nats is about 95 to 5 between the two."""
import json
import numpy as np
G=[json.loads(l) for l in open('lfm3-sum-v1-general.jsonl')]
R=[json.loads(l) for l in open('lfm3-sum-v1-rows.jsonl')]
g=np.array([r['m_self'] for r in G]); d=np.array([r['m_self'] for r in R])
print('random words in his thinking',len(g),'| decisive digit spots',len(d))
for t,name in ((0.69,'closer than 2 to 1'),(1.39,'closer than 4 to 1'),(2.94,'closer than 95 to 5')):
    print(f"  top two {name}: random words {(g<t).mean()*100:.1f}%, decisive digits {(d<t).mean()*100:.1f}%")
print(f"  median gap: random words {np.median(g):.2f} nats, decisive digits {np.median(d):.2f} nats")
