#!/usr/bin/env python3
"""How far each cortical receptor moves from rest under each of the engine's drugs: the log ratio the brain's receptors
would read (engine _lr, 0 at rest). Each drug at dose 1 on an average egg at 8 am, read after 2 hours.

usage: drug_range.py <engine folder> <out.json>
"""
import json, os, sys

sys.path.insert(0, os.path.abspath(sys.argv[1]))
import engine as chem

CORTEX = ["a2a_pfc", "b1_pfc", "a1_pfc", "d1_pfc", "m1", "ht1a_post", "ht2a", "h1", "ox2r", "gaba_a"]


def reading(e):
    return {r: round(e._lr(r), 4) for r in CORTEX if r in e.cal["rest"]}


rest = chem.Chem(hour=8.0)
rows = {"rest": reading(rest)}
for drug in chem.DRUGS:
    e = chem.Chem(hour=8.0)
    e.give(drug, 1.0)
    for _ in range(120):
        e.step(1 / 60)
    rows[drug] = reading(e)
    print(drug, rows[drug], flush=True)
json.dump(rows, open(sys.argv[2], "w"), indent=1)
