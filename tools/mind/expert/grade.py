#!/usr/bin/env python3
"""Grade expert-route runs: right/wrong from the reply's last "Answer:" line, plus thinking length. A pointer only
(audit-by-reading rule): every reply still gets read and its route tagged by hand.

Quadratics are checked loosely: integer/fraction roots must all appear; irrational or complex roots count when the
reply shows the right square root (sqrt(69), √69, i√11...) or decimals within 0.01.

usage: grade.py <run.jsonl> [more runs ...]        prints a table per run; writes <run>.graded.jsonl next to each
"""
import json, re, statistics, sys
from fractions import Fraction

import sympy as sp

NUM = r"-?\d+(?:\.\d+)?(?:/\d+)?"


def answer_line(reply, keep_tex=False):
    m = re.findall(r"(?im)^\W*answer\W*[:：]\s*(.+)$", reply)
    line = (m[-1] if m else reply.strip().splitlines()[-1] if reply.strip() else "").replace("$", "")
    return line if keep_tex else line.replace("\\", "")


def nums(s):
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
    out = []
    for t in re.findall(r"-?\s?\d+(?:\.\d+)?(?:\s?/\s?\d+)?", s.replace("frac{", "").replace("}{", "/")):
        try:
            out.append(Fraction(t.replace(" ", "")))
        except (ValueError, ZeroDivisionError):
            pass
    return out


def right(it, line):
    a, cat = it["answer"], it["cat"]
    if cat == "math500":                                       # competition answers: math-verify (the MATH-500 standard)
        from math_verify import parse, verify
        got = parse("$" + line.strip().rstrip(".") + "$")
        if not got:
            got = parse(it["reply"])                            # fall back to the last \boxed{} / expression in the reply
        return bool(got) and verify(parse("$" + a + "$"), got)
    if cat == "quadratic":
        ok = True
        for r in a:
            e = sp.sympify(r)
            if e.is_rational:
                ok &= Fraction(str(e)) in nums(line)
                continue
            rad = [str(x.args[0]) for x in e.atoms(sp.Pow) if x.exp == sp.Rational(1, 2)]
            z = complex(e)
            dec = [float(n) for n in nums(line)]
            surd = all(re.search(r"(√|sqrt)\s*[({]?" + k + r"[)}]?(?!\d)", line) for k in rad)
            ok &= surd or any(abs(d - z.real) < 0.01 for d in dec) and abs(z.imag) < 1e-9
        return ok
    if cat == "system":
        x, y = map(int, re.findall(r"-?\d+", a))
        mx, my = re.search(rf"x\s*=\s*({NUM})", line), re.search(rf"y\s*=\s*({NUM})", line)
        pair = re.search(rf"\(\s*x\s*,\s*y\s*\)\s*=\s*\(\s*({NUM})\s*,\s*({NUM})\s*\)", line)     # "(x, y) = (4, 8)"
        if pair:
            return Fraction(pair[1]) == x and Fraction(pair[2]) == y
        return bool(mx and my) and Fraction(mx[1]) == x and Fraction(my[1]) == y
    want = Fraction(a)
    got = nums(line)
    return bool(got) and (got[-1] == want or want in got[-2:])


def right_reply(it, reply):
    """Grade a whole reply on what follows its last "Answer:" (on the same line or in a display block below it).
    right(it, reply) misreads short answers inside a long reply (6+9i graded wrong); answer_line() sees only "\\]"
    when the answer sits in a \\[...\\] block on the next lines. Falls back to the whole reply when no marker."""
    m = list(re.finditer(r"(?i)answer\W*[:：]", reply))
    tail = (reply[m[-1].end():][:300] if m else reply[-300:]).strip().strip("*").strip()
    if not tail:
        return False
    if it["cat"] == "math500":                                  # math-verify reads \(...\), \[...\], \boxed{} and bare text
        from math_verify import parse, verify
        want = parse("$" + it["answer"] + "$")
        bare = re.sub(r"√\s*(\w+)", r"\\sqrt{\1}", tail).replace("π", "\\pi").rstrip(".")
        return any(bool(g) and bool(verify(want, g)) for g in (parse(tail), parse("$" + bare + "$")))   # delimited, bare
    return bool(right({**it, "reply": tail}, tail))


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    by = {}
    with open(path.replace(".jsonl", ".graded.jsonl"), "w") as f:
        for r in rows:
            r["answer_line"] = answer_line(r["reply"], keep_tex=r["cat"] == "math500")
            r["right"] = bool(right(r, r["answer_line"])) and r["finish"] == "stop"
            f.write(json.dumps(r) + "\n")
            by.setdefault((r["cat"], r["kind"]), []).append(r)
    print(f"\n{path}: {len(rows)} problems")
    print(f"  {'group':22} {'right':>9} {'think tokens (median)':>22} {'reply tokens':>13}")
    for (cat, kind), rs in sorted(by.items()):
        print(f"  {cat + ' ' + kind:22} {sum(r['right'] for r in rs):>4}/{len(rs):<4} "
              f"{statistics.median(r['think_tokens'] for r in rs):>22.0f} {statistics.median(r['reply_tokens'] for r in rs):>13.0f}")
    for kind in ("shortcut", "trap"):
        rs = [r for r in rows if r["kind"] == kind]
        if rs:
            print(f"  {'ALL ' + kind:22} {sum(r['right'] for r in rs):>4}/{len(rs):<4} "
                  f"{statistics.median(r['think_tokens'] for r in rs):>22.0f} {statistics.median(r['reply_tokens'] for r in rs):>13.0f}")
