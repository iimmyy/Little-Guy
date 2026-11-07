#!/usr/bin/env python3
"""No bar. Calm wrong on the same footing, wrong side against no side. usage: carry_extra.py <report>"""
import re, sys
import numpy as np
rows = [l.split() for l in open(sys.argv[1]).read().split("step codes A B C:\n")[1].strip().splitlines()]
A, B, C = [np.array([r[i] for r in rows]) for i in (1, 2, 3)]
wrong = lambda x: np.isin(x, ["W", "O", "F"])
doubt = lambda x: np.isin(x, ["DR", "DW", "DO", "DF", "D"])
print(f"calm wrong value: A {wrong(A).mean()*100:.1f}%, B {wrong(B).mean()*100:.1f}%, C {wrong(C).mean()*100:.1f}%")
print(f"any doubt: A {doubt(A).mean()*100:.1f}%, B {doubt(B).mean()*100:.1f}%, C {doubt(C).mean()*100:.1f}%")
print(f"step-level, C calm wrong where A was not: {int((wrong(C) & ~wrong(A)).sum())}; A calm wrong where C was not: {int((wrong(A) & ~wrong(C)).sum())}")
print(f"step-level, B calm wrong where A was not: {int((wrong(B) & ~wrong(A)).sum())}; A calm wrong where B was not: {int((wrong(A) & ~wrong(B)).sum())}")
