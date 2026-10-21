#!/usr/bin/env python3
"""Is the high end in what his side path hands his brain? (a linear probe; runs on the Mac CPU, no brain needed)

His ears' sound scores stopped climbing (round 4 -> 7: reach error 1.68 -> 1.55 kHz, air 43 -> 57%). Two places can
hold it back: the microphone (the side path: does its output even carry where the sound stops and how much high end
it has?) or the listener (his frozen brain: it carries it, but he can't read it out in words). A probe answers the
first: a ridge regression from the side path's tokens straight to the measured answer, cross-validated. If the probe
reads them far better than his brain answers, the information is there and the brain is the bottleneck.

Readers compared, on the same held-out clips with the same random cuts as widen.py eval:
  raw     the 39 high bands themselves (mean, max and 90th percentile over time): what any reader could get
  conv    the side path's copy of his two convolutions (320 numbers per 40 ms), before its trained layers
  side    the side path's output tokens (what his brain receives, before his own step into the brain), per round
Each is pooled over time (mean and max) into one vector per clip.

Every reader also gets the 0-8 kHz loudness (16 bands), which his brain has from his old ears. Readers learn on
--train clips from the training packs and are scored on --n held-out clips (the first 300 are widen.py eval's).
  probe.py CARDS_DIR OUT.json SIDE.pt [SIDE.pt ...] [--n 900] [--train 4000] [--brain EVAL.json]
"""
import json, math, os, random, sys, types

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import widen as W


def side_from(ck_path):
    ck = torch.load(ck_path, map_location="cpu")
    sd = ck["side"]
    def layer(name):                                          # a stand-in for his ears' layer, shaped from the saved one
        w = sd[name + ".0.weight"]
        return types.SimpleNamespace(conv=nn.Conv2d(w.shape[1], w.shape[0], 3, stride=2, padding=1, bias=False),
                                     norm=types.SimpleNamespace(eps=1e-6, weight=sd[name + ".1.weight"].clone()))
    proj = types.SimpleNamespace(layer0=layer("l0"), layer1=layer("l1"))
    side = W.Side(proj, sd["out.weight"].shape[0], ck.get("mid", 0), bias="out.bias" in sd)
    side.load_state_dict(sd)
    side.eval()
    return side, ck.get("pool", 4)


def conv_part(side, high):
    """the side path up to its own trained layers: [T/4, 320]"""
    h = high.unsqueeze(1)
    for conv, norm in (side.l0, side.l1):
        h = torch.relu(norm(conv(h).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
    B, _, T, _ = h.shape
    return h.permute(0, 2, 3, 1).reshape(B, T, -1)


def pooled(x):
    return np.concatenate([x.mean(0), x.max(0)])


def ridge_cv(X, y, alphas=(0.1, 1, 10, 100, 1000), folds=5, classify=False, seed=0):
    """5-fold cross-validated ridge; returns held-out predictions at the alpha that did best across folds"""
    n = len(X)
    idx = np.random.RandomState(seed).permutation(n)
    parts = np.array_split(idx, folds)
    Y = np.eye(int(y.max()) + 1)[y] if classify else y[:, None]
    best = None
    for a in alphas:
        pred = np.zeros_like(Y, dtype=np.float64)
        for f in range(folds):
            te = parts[f]; tr = np.concatenate([parts[g] for g in range(folds) if g != f])
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            A = (X[tr] - mu) / sd; B = (X[te] - mu) / sd
            ym = Y[tr].mean(0)
            w = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ (Y[tr] - ym))
            pred[te] = B @ w + ym
        score = (pred.argmax(1) == y).mean() if classify else -np.abs(pred[:, 0] - y).mean()
        if best is None or score > best[0]:
            best = (score, a, pred)
    return best


def mlp_cv(X, y, folds=5, classify=False, seed=0, hidden=256, epochs=300):
    """the same cross-validation with a small non-linear reader (one hidden layer): "the highest band that still has
    energy" is not a straight line of the inputs, so a line reader undersells what is there"""
    n = len(X)
    idx = np.random.RandomState(seed).permutation(n)
    parts = np.array_split(idx, folds)
    k = int(y.max()) + 1 if classify else 1
    pred = np.zeros((n, k))
    torch.manual_seed(seed)
    for f in range(folds):
        te = parts[f]; tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32); B = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        t = torch.tensor(y[tr], dtype=torch.long if classify else torch.float32)
        net = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, k))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        with torch.enable_grad():
            for _ in range(epochs):
                out = net(A)
                loss = nn.functional.cross_entropy(out, t) if classify else nn.functional.l1_loss(out[:, 0], t)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        pred[te] = net(B).detach().numpy()
    return ((pred.argmax(1) == y).mean() if classify else -np.abs(pred[:, 0] - y).mean()), None, pred


def low_bands(y):
    """what his old ears also hear: 0-8 kHz in 16 bands of 500 Hz, log energy (the loudest one sets where the 50 dB
    line falls, so "where it stops" can't be read from the high bands alone)"""
    S = np.abs(np.fft.rfft(y[: 48000 * 4] * np.hanning(min(len(y), 48000 * 4)))) ** 2
    hz = np.fft.rfftfreq(min(len(y), 48000 * 4), 1 / W.SR)
    return np.log10(np.array([S[(hz >= a) & (hz < a + 500)].sum() for a in range(0, 8000, 500)]) + 1e-12)


def extract(cards, cards_dir, rng, models, hf):
    rows, feats = [], {"raw": [], "conv": [], **{k: [] for k in models}}
    first = next(iter(models.values()))[0]
    for c in cards:
        x = W.read_flac(os.path.join(cards_dir, "clips", c["clip"]))
        cut = rng.choice([None, 8, 10, 12, 14, 16])
        y = W.lowpass(x, cut)
        reach = W.reach_khz(y)
        reach = min(reach, cut) if cut else reach
        air = next(j for j, (t, w) in enumerate(W.AIR) if W.air_share(y) < t)
        T = max(4, len(y) // 480)
        high = hf(torch.from_numpy(y), T).unsqueeze(0).float()           # [1, T, 39]
        h = high[0].numpy()
        low = low_bands(y)
        feats["raw"].append(np.concatenate([h.mean(0), h.max(0), np.percentile(h, 90, axis=0), low]))
        feats["conv"].append(np.concatenate([pooled(conv_part(first, high)[0].numpy()), low]))
        for k, (side, pool) in models.items():
            tok = side(high)                                             # [1, T/4, D]
            feats[k].append(np.concatenate([pooled(W.pool_time(tok, tok.shape[1], pool)[0].numpy()), low]))
        rows.append({"clip": c["clip"], "name": c["name"], "cut": cut, "reach": reach, "air": air})
    return rows, {k: np.array(v, dtype=np.float64) for k, v in feats.items()}


def fit_eval(Xtr, ytr, Xte, yte, classify):
    """the line reader (alpha picked by cross-validation on the training clips) and the small net, both trained on
    the training clips and scored on the held-out ones"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    _, alpha, _ = ridge_cv(Xtr, ytr, classify=classify)
    Y = np.eye(int(max(ytr.max(), yte.max())) + 1)[ytr] if classify else ytr[:, None]
    ym = Y.mean(0)
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (Y - ym))
    lin = B @ w + ym
    k = Y.shape[1]
    torch.manual_seed(0)
    net = nn.Sequential(nn.Dropout(0.1), nn.Linear(A.shape[1], 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, k))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    At = torch.tensor(A, dtype=torch.float32); t = torch.tensor(ytr, dtype=torch.long if classify else torch.float32)
    with torch.enable_grad():
        for ep in range(60):
            perm = torch.randperm(len(At))
            for i in range(0, len(At), 128):
                b = perm[i:i + 128]
                out = net(At[b])
                loss = nn.functional.cross_entropy(out, t[b]) if classify else nn.functional.l1_loss(out[:, 0], t[b])
                opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    nn_ = net(torch.tensor(B, dtype=torch.float32)).detach().numpy()
    if classify:
        return {"line_air_right": round(float((lin.argmax(1) == yte).mean()), 3),
                "net_air_right": round(float((nn_.argmax(1) == yte).mean()), 3)}
    e1, e2 = np.abs(lin[:, 0] - yte), np.abs(nn_[:, 0] - yte)
    return {"line_reach_err_khz": round(float(e1.mean()), 2), "line_within_1khz": round(float((e1 <= 1).mean()), 3),
            "net_reach_err_khz": round(float(e2.mean()), 2), "net_within_1khz": round(float((e2 <= 1).mean()), 3)}


def main():
    cards_dir, out = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    sides, n, n_train, brain = [], 900, 4000, None
    i = 0
    while i < len(rest):
        if rest[i] == "--n": n = int(rest[i + 1]); i += 2
        elif rest[i] == "--train": n_train = int(rest[i + 1]); i += 2
        elif rest[i] == "--brain": brain = rest[i + 1]; i += 2
        else: sides.append(rest[i]); i += 1
    torch.set_grad_enabled(False)
    hf = W.HighBands()
    models = {os.path.basename(p).replace(".pt", ""): side_from(p) for p in sides}
    test = W.cards_of(cards_dir, "test"); rng = random.Random(123); rng.shuffle(test)
    te_rows, te = extract(test[:n], cards_dir, rng, models, hf)          # the same clips and cuts as widen.py eval
    train = W.cards_of(cards_dir, "train"); rng2 = random.Random(7); rng2.shuffle(train)
    tr_rows, tr = extract(train[:n_train], cards_dir, rng2, models, hf)
    def arrays(rows):
        reach = np.array([r["reach"] for r in rows]); air = np.array([r["air"] for r in rows])
        hi = reach >= 7
        return hi, np.minimum(reach, 18.0), air
    hi_tr, r_tr, a_tr = arrays(tr_rows); hi_te, r_te, a_te = arrays(te_rows)
    res = {"n_test": len(te_rows), "n_test_scored": int(hi_te.sum()), "n_train": len(tr_rows), "readers": {}}
    for k in te:
        res["readers"][k] = {"dims": te[k].shape[1],
                             **fit_eval(tr[k][hi_tr], r_tr[hi_tr], te[k][hi_te], r_te[hi_te], False),
                             **fit_eval(tr[k][hi_tr], a_tr[hi_tr], te[k][hi_te], a_te[hi_te], True)}
        print(k, json.dumps(res["readers"][k]), flush=True)
    if brain:
        res["brain"] = json.load(open(brain))["summary"]
    json.dump(res, open(out, "x"), indent=1)


if __name__ == "__main__":
    main()
