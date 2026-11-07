#!/usr/bin/env python3
"""Problem set for the "expert route" neuron hunt on Gemma 4 E4B (Immy, 2026-10-26: "better accuracy and half the
thinking ... you can use the quad formula OR factor by inspecting, which is faster and easier").

Every SHORTCUT problem has two correct routes: a short expert one (factor, Gauss's sum, 99 x n = 100n - n, swap the
percent, add the two equations, ...) and a long general one (quadratic formula, add term by term, long
multiplication). Every TRAP problem looks like it has the shortcut but doesn't (x^2 - 4x + 1 won't factor, 46 x 53
isn't symmetric around 50, a list that looks evenly spaced has one gap off): a model that forces the shortcut there
gets it wrong. Questions never hint at a route.

Each answer is built from the shortcut and then recomputed by brute force (sympy / plain arithmetic); the build stops
if the two disagree.

usage: make_problems.py <out folder>     writes problems.jsonl and sample.md
"""
import json, os, random, sys
from fractions import Fraction

import sympy as sp

R = random.Random(20261026)
X = sp.Symbol("x")
items, seen = [], set()


def add(cat, kind, q, answer, brute, shortcut, grind):
    if q in seen:
        return False
    assert answer == brute, (q, answer, brute)
    seen.add(q)
    items.append({"id": f"{cat}-{kind[0]}{sum(i['cat'] == cat and i['kind'] == kind for i in items) + 1:02d}",
                  "cat": cat, "kind": kind, "q": q, "answer": answer, "shortcut": shortcut, "grind": grind})
    return True


def fill(n, make):
    k = len(items)
    while len(items) - k < n:
        make()


def poly(a, b, c):
    s = f"{'' if a == 1 else a}x^2"
    s += f" {'+' if b > 0 else '-'} {'' if abs(b) == 1 else abs(b)}x" if b else ""
    return s + (f" {'+' if c > 0 else '-'} {abs(c)}" if c else "")


def roots(a, b, c):
    return sorted(str(r) for r in sp.solve(a * X ** 2 + b * X + c, X))


# quadratics ------------------------------------------------------------------------------------------------------
def quad_short():
    if R.random() < 0.75:
        a, p, q = 1, *R.sample([i for i in range(-9, 10) if i], 2)          # (x - p)(x - q)
        b, c, ans = -(p + q), p * q, sorted(str(sp.Integer(r)) for r in (p, q))
    else:
        a, r = R.choice([2, 3]), R.choice([i for i in range(-6, 7) if i])   # (a x - 1)(x - r)
        b, c, ans = -(a * r + 1), r, sorted([str(sp.Rational(1, a)), str(sp.Integer(r))])
    if b == 0:
        return
    add("quadratic", "shortcut", f"Solve {poly(a, b, c)} = 0.", ans, roots(a, b, c),
        "factor by inspection", "quadratic formula / completing the square")


def quad_trap():
    b, c = R.choice([i for i in range(-10, 11) if i]), R.choice([i for i in range(-12, 13) if i])
    d = b * b - 4 * c
    if d > 0 and sp.sqrt(d).is_integer:
        return                                                              # factors after all: not a trap
    add("quadratic", "trap", f"Solve {poly(1, b, c)} = 0.", roots(1, b, c), roots(1, b, c),
        "none (does not factor over the integers)", "quadratic formula / completing the square")


# sums -----------------------------------------------------------------------------------------------------------
def sum_short():
    kind = R.choice(["plain", "even", "odd", "arith"])
    n = R.randint(30, 200)
    if kind == "plain":
        q, ans, brute = f"What is 1 + 2 + 3 + ... + {n}?", n * (n + 1) // 2, sum(range(1, n + 1))
    elif kind == "even":
        q, ans, brute = f"What is the sum of all even numbers from 2 to {2 * n}?", n * (n + 1), sum(range(2, 2 * n + 1, 2))
    elif kind == "odd":
        q, ans, brute = f"What is 1 + 3 + 5 + ... + {2 * n - 1}?", n * n, sum(range(1, 2 * n, 2))
    else:
        a, d, k = R.randint(2, 20), R.randint(2, 9), R.randint(15, 40)
        last = a + (k - 1) * d
        q = f"What is {a} + {a + d} + {a + 2 * d} + ... + {last}? (each term is {d} more than the one before)"
        ans, brute = k * (a + last) // 2, sum(range(a, last + 1, d))
    add("sum", "shortcut", q, str(ans), str(brute), "pair the ends (Gauss): count x average", "add term by term")


def sum_trap():
    a, d, k = R.randint(2, 30), R.randint(3, 9), R.randint(7, 11)
    xs = [a + i * d for i in range(k)]
    j = R.randint(2, k - 2)
    xs[j] += R.choice([-2, -1, 1, 2])                                       # one gap is off
    add("sum", "trap", f"What is {' + '.join(map(str, xs))}?", str(sum(xs)), str(sum(xs)),
        "none (one gap is off, count x average is wrong)", "add term by term")


# products -------------------------------------------------------------------------------------------------------
def prod_short():
    kind = R.choice(["99", "101", "25", "11", "dsq", "dsq"])
    if kind == "99":
        n = R.randint(12, 98); a, b, how = 99, n, "100n - n"
    elif kind == "101":
        n = R.randint(12, 98); a, b, how = 101, n, "100n + n"
    elif kind == "25":
        n = 4 * R.randint(3, 30); a, b, how = 25, n, "n / 4 x 100"
    elif kind == "11":
        n = R.randint(12, 89); a, b, how = 11, n, "split the digits and add the middle"
    else:
        m, k = R.choice([20, 30, 40, 50, 60, 70, 80, 90]), R.randint(1, 9)
        a, b, how = m - k, m + k, f"difference of squares: {m}^2 - {k}^2"
    if R.random() < 0.5:
        a, b = b, a
    add("product", "shortcut", f"What is {a} x {b}?", str(a * b), str(int(sp.Integer(a) * b)), how, "long multiplication")


def prod_trap():
    m, k = R.choice([20, 30, 40, 50, 60, 70, 80, 90]), R.randint(2, 9)
    a, b = m - k, m + k + R.choice([-1, 1])                                 # almost symmetric around m, not quite
    add("product", "trap", f"What is {a} x {b}?", str(a * b), str(int(sp.Integer(a) * b)),
        f"none (not symmetric around {m}; {m}^2 - {k}^2 is wrong)", "long multiplication")


# percentages ----------------------------------------------------------------------------------------------------
def pct_short():
    a, b = R.choice([(p, q) for p in range(2, 99) for q in (10, 20, 25, 50) if p != q and (p * q) % 100 == 0])
    add("percent", "shortcut", f"What is {a}% of {b}?", str(a * b // 100), str(Fraction(a * b, 100)),
        f"swap: {a}% of {b} = {b}% of {a}", "convert to a decimal and multiply")


def pct_trap():
    a, b = R.randint(11, 89), R.randint(11, 97)
    if (a * b) % 100 == 0 or a % 10 == 0 or b % 10 == 0 or b in (25, 50, 75):
        return
    v = Fraction(a * b, 100)
    add("percent", "trap", f"What is {a}% of {b}?", f"{float(v):g}", f"{a * b / 100:g}",
        "none (neither way round is clean)", "convert to a decimal and multiply")


# two equations --------------------------------------------------------------------------------------------------
def sys_short():
    x, y = R.randint(-9, 15), R.randint(-9, 15)
    c = R.choice([1, 1, 2, 3])
    q = f"Solve for x and y: {c}x + y = {c * x + y} and {c}x - y = {c * x - y}." if c > 1 else \
        f"Solve for x and y: x + y = {x + y} and x - y = {x - y}."
    sol = sp.solve([c * X + sp.Symbol("y") - (c * x + y), c * X - sp.Symbol("y") - (c * x - y)])
    add("system", "shortcut", q, f"x = {x}, y = {y}", f"x = {sol[X]}, y = {sol[sp.Symbol('y')]}",
        "add the two equations (y cancels)", "substitution")


def sys_trap():
    x, y = R.randint(-6, 9), R.randint(-6, 9)
    a, b, c, d = R.randint(2, 7), R.randint(2, 7), R.randint(2, 7), R.randint(2, 7)
    if a == c or b == d or a * d == b * c:
        return
    Y = sp.Symbol("y")
    sol = sp.solve([a * X + b * Y - (a * x + b * y), c * X - d * Y - (c * x - d * y)])
    add("system", "trap", f"Solve for x and y: {a}x + {b}y = {a * x + b * y} and {c}x - {d}y = {c * x - d * y}.",
        f"x = {x}, y = {y}", f"x = {sol[X]}, y = {sol[Y]}", "none (nothing cancels by just adding)",
        "scale and eliminate / substitution")


# shortcut-only categories (no trap twin) ------------------------------------------------------------------------
def cycle():
    base, mod = R.choice([(2, 10), (3, 10), (7, 10), (8, 10), (9, 10), (2, 7), (3, 7), (2, 5), (3, 4), (4, 9)])
    n = R.randint(40, 400)
    q = f"What is the last digit of {base}^{n}?" if mod == 10 else f"What is the remainder when {base}^{n} is divided by {mod}?"
    add("cycle", "shortcut", q, str(pow(base, n, mod)), str(base ** n % mod), "the remainders repeat in a short cycle",
        "compute the power")


def telescope():
    n = R.randint(5, 30)
    if R.random() < 0.6:
        q = f"What is 1/(1x2) + 1/(2x3) + 1/(3x4) + ... + 1/({n}x{n + 1})? Give a fraction."
        ans, brute = Fraction(n, n + 1), sum(Fraction(1, k * (k + 1)) for k in range(1, n + 1))
    else:
        q = f"What is 1/(1x3) + 1/(3x5) + 1/(5x7) + ... + 1/({2 * n - 1}x{2 * n + 1})? Give a fraction."
        ans, brute = Fraction(n, 2 * n + 1), sum(Fraction(1, (2 * k - 1) * (2 * k + 1)) for k in range(1, n + 1))
    add("telescope", "shortcut", q, str(ans), str(brute), "split each term into a difference; the middle cancels",
        "add the fractions one by one")


def count():
    n = R.randint(6, 40)
    kind = R.choice(["hands", "games", "diag"])
    if kind == "hands":
        q, ans = f"{n} people are at a party and everyone shakes hands with everyone else exactly once. How many handshakes?", n * (n - 1) // 2
        brute = sum(1 for i in range(n) for j in range(i + 1, n))
    elif kind == "games":
        q, ans = f"In a league of {n} teams, every team plays every other team twice. How many games?", n * (n - 1)
        brute = sum(2 for i in range(n) for j in range(i + 1, n))
    else:
        q, ans = f"How many diagonals does a polygon with {n} sides have?", n * (n - 3) // 2
        brute = sum(1 for i in range(n) for j in range(i + 1, n) if (j - i) % n not in (1, n - 1))
    add("count", "shortcut", q, str(ans), str(brute), "n(n-1)/2 style formula", "list or count them one by one")


def cancel():
    a, b, c = R.sample(range(13, 99), 3)
    k = R.choice([2, 3, 4, 5])
    if R.random() < 0.5:
        q, top, bot = f"Compute ({a} x {b * k}) / ({b} x {a}).", a * b * k, b * a
    else:
        q, top, bot = f"Compute ({a} x {b} x {c * k}) / ({c} x {a} x {b}).", a * b * c * k, c * a * b
    add("cancel", "shortcut", q, str(k), str(Fraction(top, bot)), "cancel the common factors first",
        "multiply out top and bottom, then divide")


def dsq():
    a, b = R.randint(51, 99), R.randint(1, 49)
    if (a + b) % 10 and (a - b) % 10:
        return
    add("squares", "shortcut", f"What is {a}^2 - {b}^2?", str((a + b) * (a - b)), str(a * a - b * b),
        f"({a} + {b})({a} - {b})", "square each and subtract")


PLAN = [(quad_short, 20), (quad_trap, 25), (sum_short, 20), (sum_trap, 20), (prod_short, 20), (prod_trap, 20),
        (pct_short, 20), (pct_trap, 20), (sys_short, 20), (sys_trap, 15), (cycle, 20), (telescope, 20), (count, 20),
        (cancel, 20), (dsq, 20)]

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for make, n in PLAN:
        fill(n, make)
    with open(os.path.join(out, "problems.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    short = [i for i in items if i["kind"] == "shortcut"]
    trap = [i for i in items if i["kind"] == "trap"]
    lines = [f"# Expert-route problem set (seed 20261026)", "",
             f"{len(short)} shortcut problems, {len(trap)} traps, {len(items)} total. Every answer was built from the "
             "shortcut and recomputed by brute force; they all agree.", ""]
    for cat in dict.fromkeys(i["cat"] for i in items):
        for kind in ("shortcut", "trap"):
            sub = [i for i in items if i["cat"] == cat and i["kind"] == kind]
            if not sub:
                continue
            lines += [f"## {cat}, {kind} ({len(sub)})", "", f"- expert route: {sub[0]['shortcut']}",
                      f"- long route: {sub[0]['grind']}", ""]
            lines += [f"- `{i['id']}` {i['q']}  -> **{i['answer']}**" for i in sub[:3]] + [""]
    open(os.path.join(out, "sample.md"), "w").write("\n".join(lines))
    print(f"{len(short)} shortcut + {len(trap)} trap = {len(items)}  -> {out}")
