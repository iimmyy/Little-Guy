#!/usr/bin/env python3
"""The open caveat of check 3 (2026-11-03): the scene error readout (AUROC ~0.9 at index 14) was learned on SWAPPED
numbers, which contradict his own earlier lines. A natural mistake does not: his wrong number is what his own (wrong)
work led to. Does the readout still see it?
Material: the untouched model's own night runs (dev set, never the sealed fresh set; greedy + sampled seeds, the same
problems several times), the finished runs with a whole-number true answer A and a whole-number final answer. The fork
is fit_test.py's: cut after the last sentence in his thinking that states his final answer as a result (A if he was
right, his wrong W if not). His stream at the fork's last word, every depth (untouched model, the same prompt the runs
used). Problems he got both right and wrong in different runs give a same-problem comparison (difficulty held fixed).
Finished whole-number wrongs turned out to be nearly absent (1 of 167 finished whole-number runs), so the runs that
ran out of words after stating the RIGHT answer are captured too: the trust side. Kinds: right handed in, wrong handed
in, right then kept checking until cut off.
Read mode: the error readout is fit on the whole fit test (story problems, swaps, target |log(said / true)|) and applied
unchanged to the night forks. If a right answer he then kept doubting reads as "fits" like a right answer he handed in,
the scene already says "it holds" and he does not use it (the trust side is the lever). Caveat: cut-off forks sit deep
in long thinking, handed-in forks near the end of shorter thinking.
usage: natural_error.py <out.npz> <runs.jsonl> [<runs.jsonl> ...] [hf= dev=cuda]
       natural_error.py read <out.npz> <fit-test.npz> <out.txt>
"""
import json, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from situation_probe import TAIL
from fit_test import fork_of

KINDS = ("right, handed in", "wrong, handed in", "right, then kept checking until cut off")


def whole(s):
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return None
    return int(v) if v == int(v) else None


def natural_forks(run_files):
    items = []
    for f in run_files:
        for t in map(json.loads, open(f)):
            A = whole(str(t["answer"]))
            if A is None or A < 2:
                continue
            if t["finish"] != "stop":                     # ran out of words: did he state the right answer first?
                fk = fork_of(t["thinking"], A)
                if fk:
                    items.append((t["id"], t["q"], t["thinking"], A, A, fk, os.path.basename(f), 2))
                continue
            if "Answer:" not in t["reply"]:
                continue
            got = re.findall(r"-?\d[\d,]*(?:\.\d+)?", t["reply"].split("Answer:")[-1])[:1]
            W = whole(got[0]) if got else None
            if W is None or W < 1:
                continue
            fk = fork_of(t["thinking"], W)
            if fk:
                items.append((t["id"], t["q"], t["thinking"], A, W, fk, os.path.basename(f), int(W != A)))
    return items


def capture(out, run_files, opt):
    import torch
    from patients import _lfm3
    dev = opt.get("dev", "cuda")
    items = natural_forks(run_files)
    print(f"{len(items)} natural forks: " + ", ".join(f"{nm} {sum(1 for x in items if x[7] == k)}" for k, nm in enumerate(KINDS)), flush=True)
    p_ = _lfm3(dev, opt.get("hf"))
    tok, m = p_.tok, p_.model
    cap = []
    hs = [l.register_forward_hook(lambda mod, a, o: cap.append((o[0] if isinstance(o, tuple) else o)[0])) for l in m.model.layers]
    hs.append(m.model.embed_tokens.register_forward_hook(lambda mod, a, o: cap.insert(0, o[0])))
    X = []
    for k, (iid, text, think, A, W, (s0, s1, cut), run, kind) in enumerate(items):
        head = tok.apply_chat_template([{"role": "user", "content": text + TAIL}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) + "\n"
        ids = tok(head + think[:cut], add_special_tokens=False).input_ids
        cap.clear()
        with torch.no_grad():
            m.model(input_ids=torch.tensor([ids], device=dev))
        X.append(torch.stack([c_[-1] for c_ in cap]).float().cpu().numpy().astype(np.float16))
        if k % 50 == 0:
            print(f"{k}/{len(items)}", flush=True)
    np.savez(out, X=np.stack(X), ids=np.array([x[0] for x in items]), A=np.array([x[3] for x in items]),
             W=np.array([x[4] for x in items]), run=np.array([x[6] for x in items]), kind=np.array([x[7] for x in items]))
    print("saved", out)


def fit_pred(Xa, ya, Xb, al):
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    Aa, Bb = (Xa - mu) / sd, (Xb - mu) / sd
    m = ya.mean()
    c = np.linalg.solve(Aa @ Aa.T + al * np.eye(len(Aa)), ya - m)
    return (Bb @ Aa.T) @ c + m


def auroc(pos, neg):
    allv = np.concatenate([pos, neg]); r = np.empty(len(allv)); r[np.argsort(allv)] = np.arange(1, len(allv) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def read(nat_f, fit_f, out):
    nz, fz = np.load(nat_f), np.load(fit_f, allow_pickle=True)
    XN, AN, ids, kind = nz["X"], nz["A"].astype(float), nz["ids"], nz["kind"]
    XF, A, B, C = fz["X"], fz["A"].astype(float), fz["b"].astype(float), fz["c"].astype(float)
    n = len(A)
    err = np.abs(np.log(np.stack([A, B, C], 1) / A[:, None])).reshape(-1)
    fold = np.random.RandomState(0).permutation(n) % 5
    F = np.repeat(fold, 3)
    hand, cut = kind == 0, kind == 2
    both = sorted({i for i in set(ids[cut]) if (ids[hand] == i).any()})
    L = [f"natural forks vs the swap-trained error readout (fit test, story problems): "
         + ", ".join(f"{nm} {int((kind == k).sum())}" for k, nm in enumerate(KINDS))
         + f"; {len(both)} problems both handed in right and cut off after stating the right answer"]
    for li in range(4, XF.shape[2], 2):
        Xf = XF[:, :, li].reshape(n * 3, -1).astype(np.float64)
        al = 1e3
        p = fit_pred(Xf, err, XN[:, li].astype(np.float64), al)
        ref = np.zeros((n, 3))
        for f in range(5):
            tr, te = F != f, fold == f
            for v in range(3):
                ref[te, v] = fit_pred(Xf[tr], err[tr], XF[te, v, li].astype(np.float64), al)
        row = f"  index {li:2d}: readout median (0 = fits, ~0.7 = a swap's typical error) | story fit test held out: right {np.median(ref[:, 0]):.3f} swaps {np.median(ref[:, 1:]):.3f}"
        row += " | night: " + ", ".join(f"{KINDS[k].split(',')[0] + KINDS[k].split(',')[1][:8]} {np.median(p[kind == k]):.3f}" for k in range(3) if (kind == k).any())
        if hand.any() and cut.any():
            row += f" | AUROC cut-off-right vs handed-in-right {auroc(p[cut], p[hand]):.2f}"
            if both:
                g = np.array([p[(ids == i) & cut].mean() - p[(ids == i) & hand].mean() for i in both])
                row += f"; same problem: cut-off reads higher in {np.mean(g > 0):.0%} of {len(both)}"
        L.append(row)
        print(row, flush=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "read":
        read(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        opt = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
        capture(sys.argv[1], [a for a in sys.argv[2:] if "=" not in a], opt)
