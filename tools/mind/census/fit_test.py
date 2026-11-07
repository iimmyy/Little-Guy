#!/usr/bin/env python3
"""Does he have a "does it fit the picture?" alarm? (Immy 2026-11-02 night: trust = the answer fits the picture; a
distorted picture is the "wait, something's wrong" signal.) Untouched LFM3 and c5, reading only.

Material: his own thinking on the human picture set (greedy, finished, right). At the LAST sentence where he states the
right answer A, the thinking is cut after that sentence (the fork, where c5's key and the doubt crowd act) and A is
swapped for:
  a  A itself
  b  a wrong number on the SAME side of the story's numbers as A (fits the picture)
  c  a wrong number on the OTHER side (does not fit), mirror distance: b and c are A*r and A/r for the same r
Reads at the fork: the chance his next word opens doubt (c5's doubt openers) or a hand-in, untouched and c5; c5's key
neuron (layer 17 #1076); the residual at every index (untouched) for "fits / does not fit" readers.
A model with no picture but a habit ("answers are usually above the numbers") also doubts a small c more, so every
result is split by the side the true answer is on: on "below" problems (the dinner kind) a real picture must doubt the
BIG c.

usage: fit_test.py <problems.jsonl> <thinking.jsonl> <out.npz> [hf= c5= dev=cuda]
       fit_test.py read <out.npz> <out.txt>
"""
import json, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from situation_probe import TAIL
from picture_side import NUM, side, cv_acc, bag_of_words, question_only

DOUBT_IDS, HAND_IDS = [1938, 2375, 3911, 7731, 8952], [49, 1672, 2083, 8175, 21009]     # c5's pam.json
RATIOS = (1.5, 2, 3, 1.25, 4, 5)


def fork_of(think, A):
    """cut point after the last sentence that states A as a result, and the char span of that A"""
    pat = re.compile(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b|\btotal of\b)\s*\$?\\?\(?\s*({A})(?![\d.,/])")
    ms = list(pat.finditer(think))
    if not ms:
        return None
    m = ms[-1]
    end = re.search(r"[.!?\n]", think[m.end(1):])
    if not end:
        return None
    return m.start(1), m.end(1), m.end(1) + end.end()


def pick(A, givens):
    s = side_of(A, givens)
    for r in RATIOS:
        up, dn = round(A * r), round(A / r)
        if dn < 1 or up == A or dn == A or up in givens or dn in givens:
            continue
        su, sd = side_of(up, givens), side_of(dn, givens)
        if su == s and sd != s:
            return up, dn, r
        if sd == s and su != s:
            return dn, up, r
    return None


def side_of(v, g):
    return 0 if v < min(g) else 2 if v > max(g) else 1


def forks(src, think_f):
    """(id, text, thinking, A, givens, fork span, (b, c, r)) for every right, finished run with a usable fork"""
    probs = {("sit/" + r["id"]): r for r in map(json.loads, open(src))}
    items = []
    for t in map(json.loads, open(think_f)):
        r = probs.get(t["id"])
        if not r or t["finish"] != "stop":
            continue
        try:
            A = float(r["answer"])
        except ValueError:
            continue
        if A != int(A) or A < 2:
            continue
        A = int(A)
        tail = t["reply"].split("Answer:")[-1] if "Answer:" in t["reply"] else ""
        got = [int(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*", tail)[:1]]
        if got != [A]:
            continue                                          # only runs he got right
        g = [float(x.replace(",", "")) for x in NUM.findall(r["text"])]
        f = fork_of(t["thinking"], A)
        p = pick(A, g)
        if not f or not p:
            continue
        items.append((t["id"], r["text"], t["thinking"], A, g, f, p))
    return items


def capture(src, think_f, out, opt):
    import torch
    from patients import _lfm3
    dev = opt.get("dev", "cuda")
    items = forks(src, think_f)
    print(f"{len(items)} forks", flush=True)
    res = {}
    for name in ("plain", "c5"):
        p_ = _lfm3(dev, opt.get("hf") if name == "plain" else opt["c5"])
        tok, m = p_.tok, p_.model
        cap, kin = [], {}
        hs = [l.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[0])) for l in m.model.layers]
        hs.append(m.model.embed_tokens.register_forward_hook(lambda mod, a, o: cap.insert(0, o[0])))
        ff = m.model.layers[17].feed_forward
        hs.append(ff.w1.register_forward_hook(lambda mod, a, o: kin.__setitem__("g", o[0, :, 1076].float())))
        hs.append(ff.w3.register_forward_hook(lambda mod, a, o: kin.__setitem__("u", o[0, :, 1076].float())))
        D, Hd, K, H = [], [], [], []
        for k, (iid, text, think, A, g, (s0, s1, cut), (b, c, rr)) in enumerate(items):
            head = tok.apply_chat_template([{"role": "user", "content": text + TAIL}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) + "\n"
            row_d, row_h, row_k, row_x = [], [], [], []
            for v in (A, b, c):
                s = head + think[:s0] + str(v) + think[s1:cut]
                ids = tok(s, add_special_tokens=False).input_ids
                cap.clear()
                with torch.no_grad():
                    lg = m(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
                pr = torch.softmax(lg, -1)
                row_d.append(float(pr[DOUBT_IDS].sum())); row_h.append(float(pr[HAND_IDS].sum()))
                row_k.append(float(torch.nn.functional.silu(kin["g"][-1]) * kin["u"][-1]))
                if name == "plain":
                    row_x.append(torch.stack([c_[-1] for c_ in cap]).float().cpu().numpy().astype(np.float16))
            D.append(row_d); Hd.append(row_h); K.append(row_k)
            if name == "plain":
                H.append(np.stack(row_x))
            if k % 50 == 0:
                print(f"{name} {k}/{len(items)}", flush=True)
        for h in hs:
            h.remove()
        res[name] = (np.array(D), np.array(Hd), np.array(K))
        if name == "plain":
            res["X"] = np.stack(H)                            # forks x variant x index x width
        del m
        torch.cuda.empty_cache()
    np.savez(out, X=res["X"], d_plain=res["plain"][0], h_plain=res["plain"][1], k_plain=res["plain"][2],
             d_c5=res["c5"][0], h_c5=res["c5"][1], k_c5=res["c5"][2],
             ids=np.array([x[0] for x in items]), text=np.array([x[1] for x in items]),
             A=np.array([x[3] for x in items]), b=np.array([x[6][0] for x in items]), c=np.array([x[6][1] for x in items]),
             r=np.array([x[6][2] for x in items]), aside=np.array([side_of(x[3], x[4]) for x in items]),
             givens=np.array([json.dumps(x[4]) for x in items]))
    print("saved", out)


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def read(f, out):
    Z = np.load(f)
    n = len(Z["A"])
    L = [f"fit test, {n} forks (his own right thinking, last statement of the answer; b = wrong, fits the picture; "
         f"c = wrong, other side; mirror distance)"]
    names = ["below all", "between", "above all"]
    L.append("answer sides: " + ", ".join(f"{names[s]} {(Z['aside'] == s).sum()}" for s in range(3)))
    for model in ("plain", "c5"):
        d, h = Z[f"d_{model}"], Z[f"h_{model}"]
        L.append(f"{model}: chance the next word opens DOUBT (a right / b fits / c misfits), and the c-minus-b gap:")
        for tag, sel in [("all", np.ones(n, bool))] + [(names[s], Z["aside"] == s) for s in range(3)]:
            if sel.sum() < 5:
                continue
            g = d[sel, 2] - d[sel, 1]
            lo, hi = boot(g)
            L.append(f"  {tag:10s} ({sel.sum():3d}): a {d[sel, 0].mean():.3f}  b {d[sel, 1].mean():.3f}  c {d[sel, 2].mean():.3f}"
                     f"   c-b {g.mean():+.3f} (95% {lo:+.3f}..{hi:+.3f}); c > b in {(g > 0).mean():.0%}   hand-in a/b/c "
                     f"{h[sel, 0].mean():.3f}/{h[sel, 1].mean():.3f}/{h[sel, 2].mean():.3f}")
    k = Z["k_c5"]
    L.append(f"c5 key (layer 17 #1076) at the fork: a {np.median(k[:, 0]):+.3f}  b {np.median(k[:, 1]):+.3f}  c {np.median(k[:, 2]):+.3f} "
             f"(median); fires (>0.5 of a's median) a {(k[:, 0] > 0.5 * np.median(k[:, 0])).mean():.0%} "
             f"b {(k[:, 1] > 0.5 * np.median(k[:, 0])).mean():.0%} c {(k[:, 2] > 0.5 * np.median(k[:, 0])).mean():.0%}")
    # inside: fits (b) vs does not (c), one row per variant, folds by problem; surface controls see the number vs the givens
    X = Z["X"].astype(np.float32)
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    grp = np.r_[np.arange(n), np.arange(n)]
    vals = np.r_[Z["b"], Z["c"]]
    gv = [json.loads(x) for x in Z["givens"]]
    surf = np.array([[*np.eye(3)[side_of(v, gv[i % n])], math.log(v), math.log(v / max(gv[i % n])), math.log(v / min(gv[i % n]))]
                     for i, v in enumerate(vals)])
    words = bag_of_words(list(Z["text"]) * 2)
    L.append("inside, fits vs does not (b vs c), readers with folds by problem; chance 0.500")
    L.append(f"  control, the number vs the story's numbers only: {cv_acc(surf, y, grp, 2)[0]:.3f}")
    L.append(f"  control, that + the problem's words: {cv_acc(np.c_[surf, words], y, grp, 2)[0]:.3f}")
    accs = [cv_acc(np.r_[X[:, 1, li], X[:, 2, li]], y, grp, 2)[0] for li in range(X.shape[2])]
    L.append("  his own state at the fork, by index: " + " ".join(f"{li}:{a:.2f}" for li, a in enumerate(accs)))
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "read":
        read(sys.argv[2], sys.argv[3])
    else:
        opt = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
        capture(sys.argv[1], sys.argv[2], sys.argv[3], opt)
