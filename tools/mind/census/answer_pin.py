#!/usr/bin/env python3
"""Check 2 for Immy's scene formula (2026-11-03): does his stream hold a*, WHERE THE SCENE WANTS THE ANSWER, before he
has worked anything out? The formula reads error = pin(stated) - a*; a* has to come from his own picture.
On the picture set (846 human-written story problems, capture from picture_side.py: his stream at the question mark
and at the first thinking word, every depth), read log(answer) with a ridge line, folds by problem, strength chosen on
inner folds. Controls (what could guess the answer's size with no picture): the given numbers' sizes alone (count,
sizes in order, spread, divisibility; picture_side.number_feats), the question sentence's words, both. The real test
is the stream ADDED to the numbers control: does his state know more than the givens' sizes already tell?
Reported: R^2 of log(answer), the median miss as a factor, and the share read within x1.5 / x2 (the formula needs
misses well under the x1.5-x5 wrong answers it should catch).
usage: answer_pin.py <picture-cap.npz> <picture-set.jsonl> <out.txt>
"""
import json, math, sys
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from picture_side import number_feats, bag_of_words, question_only

z = np.load(sys.argv[1], allow_pickle=True)
ans = {r["id"]: r["answer"] for r in map(json.loads, open(sys.argv[2]))}
s = z["grp"] == "set"
ids, texts, prob = z["ids"][s], list(z["text"][s]), z["prob"][s]
a = np.array([float(ans.get(i, -1)) for i in ids])
ok = a > 0
y = np.log(a[ok])
ids, prob = ids[ok], prob[ok]
texts = [t for t, k in zip(texts, ok) if k]
QS, FT = z["QS"][s][ok], z["FT"][s][ok]
ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
ug = np.unique(prob)
fold = {g: i % 5 for i, g in enumerate(np.random.RandomState(0).permutation(ug))}
F = np.array([fold[g] for g in prob])


def fit_pred(Xa, ya, Xb, al):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    A, B = (Xa - mu) / sd, (Xb - mu) / sd
    m = ya.mean()
    c = np.linalg.solve(A @ A.T + al * np.eye(len(A)), ya - m)
    return (B @ A.T) @ c + m


def cv(X):
    pred = np.zeros(len(y))
    for f in range(5):
        tr, te = F != f, F == f
        best = None
        for al in ALPHAS:
            err = sum(((fit_pred(X[tr & (F != g)], y[tr & (F != g)], X[tr & (F == g)], al) - y[tr & (F == g)]) ** 2).sum() for g in set(F[tr]))
            if best is None or err < best[0]:
                best = (err, al)
        pred[te] = fit_pred(X[tr], y[tr], X[te], best[1])
    return pred


def score(p):
    e = np.abs(p - y)
    return (f"R^2 {1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}, median miss x{math.exp(np.median(e)):.2f}, "
            f"within x1.5 {np.mean(e < math.log(1.5)):.0%}, within x2 {np.mean(e < math.log(2)):.0%}")


NUMS = number_feats(texts)
L = [f"answer pin: {len(y)} problems (answer > 0), answers {a[ok].min():g}..{a[ok].max():g}, median {np.median(a[ok]):g}"]
L.append(f"  guess the typical size (no information): {score(np.full(len(y), y.mean()))}")
pn = cv(NUMS)
L.append(f"  control, given numbers' sizes alone: {score(pn)}")
L.append(f"  control, question words: {score(cv(bag_of_words([question_only(t) for t in texts])))}")
L.append(f"  control, question words + numbers: {score(cv(np.concatenate([bag_of_words([question_only(t) for t in texts]), NUMS], 1)))}")
print("\n".join(L), flush=True)
for name, H in (("question mark", QS), ("first thinking word", FT)):
    for li in range(2, H.shape[1], 2):
        X = H[:, li].astype(np.float64)
        ps = cv(X)
        pc = cv(np.concatenate([X, NUMS * 10], 1))
        L.append(f"  {name}, index {li:2d}: stream {score(ps)} | stream + numbers {score(pc)}")
        print(L[-1], flush=True)
open(sys.argv[3], "w").write("\n".join(L) + "\n")
print("\n".join(L))
