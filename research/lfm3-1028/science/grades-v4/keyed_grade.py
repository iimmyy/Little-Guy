#!/usr/bin/env python3
"""sci_grade2 with the keys-v2 overlay applied. Saved rows carry the old key, so a scorer that passes r["answer"] straight to grade() misses the repairs."""
import json, os
from sci_grade2 import grade
KEYS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys-v2.json")))
def repaired(i):
    for k, v in KEYS["repaired"].items():
        if i.endswith(k):
            return v
    return None
def is_unclear(i):
    return any(i.endswith(k) for k in KEYS["unclear"])
def keyed_grade(finish, reply, i, answer, scale):
    """right, wrong, sign, cut or unclear, for one attempt on problem i."""
    if finish != "stop":
        return "cut"
    if is_unclear(i):
        return "unclear"
    v = repaired(i)
    if v is None:
        return grade(dict(finish=finish, answer=answer, scale=scale, reply=reply))
    got = [grade(dict(finish=finish, answer=str(a), scale=0, reply=reply)) for a in v["answer"]]
    return "right" if "right" in got else ("sign" if "sign" in got else "wrong")
def key_text(i, answer, scale, unit):
    """What a human reader should grade against."""
    v = repaired(i)
    if v is not None:
        return f"{' or '.join(str(a) for a in v['answer'])} {v.get('unit', '')} (repaired key, was {v.get('was')}; reason: {v.get('why')})".strip()
    s = f"{answer}"
    if str(scale or 0) not in ("0", ""):
        s += f" x 10^{scale}"
    return f"{s} {unit or ''}".strip()
