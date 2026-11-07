#!/usr/bin/env python3
"""Life model v0: a tiny brain that never stops, learning his body by living it.

A recurrent network (GRU) reads his life one minute at a time: his body and his world now. Before each stretch of
life it says what his body will be 1, 15 and 60 minutes later; then the stretch happens, the error (his surprise)
is written down, and only then does he learn from it. His state is never reset. One whole life is kept aside and
never learned from, to check that he learned his body rather than memorizing his days.

Surprise is measured against two baselines a brain must beat: 'nothing changes' (the body later = the body now,
surprise 1.0 by construction) and 'same as yesterday' (the change ahead = the change at this time yesterday).
Skill = 1 - surprise: 0 means no better than 'nothing changes', 1 means perfect.

usage: learn.py <lives folder> <out folder> <seconds> [hidden]
"""
import json, os, sys, time

import numpy as np
import torch

H = [1, 15, 60]                                              # minutes ahead
CHUNK = 360                                                  # he learns after every 6 hours lived
DAY = 1440


def load(src):
    meta = json.load(open(os.path.join(src, "meta.json")))
    lives = [np.load(os.path.join(src, f"life_{k:02d}.npz")) for k in range(meta["workers"])]
    T = min(len(z["body"]) for z in lives)                  # lived in lockstep, so cut to the shortest life
    return meta, np.stack([z["body"][:T] for z in lives]), np.stack([z["world"][:T] for z in lives])


def deltas(Z):
    """the change ahead at each horizon: [lives, T - 60, horizons, channels]"""
    T = Z.shape[1] - max(H)
    return np.stack([Z[:, h:h + T] - Z[:, :T] for h in H], axis=2)


class Brain(torch.nn.Module):
    def __init__(self, n_in, n_body, hidden):
        super().__init__()
        self.core = torch.nn.GRU(n_in, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, len(H) * n_body)
        self.n_body = n_body

    def forward(self, x, h=None):
        y, h = self.core(x, h)
        return self.head(y).view(*y.shape[:2], len(H), self.n_body), h


def surprise(pred, y, scale):
    """mean squared error per horizon, in units of 'nothing changes' (1.0 = no better than that)"""
    return ((pred - y) ** 2 / scale).mean(dim=(0, 1, 3))


if __name__ == "__main__":
    src, out, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    hidden = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    meta, body, world = load(src)
    train_b, test_b = body[:-1], body[-1:]                  # the last life is never learned from
    mu, sd = train_b.mean(axis=(0, 1)), train_b.std(axis=(0, 1))
    # channels that (almost) never change over an hour teach nothing and are left out: judged by how much they
    # change, not by their spread, since a channel that jumped once has a spread but no hour-to-hour change
    Z = np.clip((body - mu) / np.maximum(sd, 1e-8), -10, 10).astype(np.float32)
    live = (deltas(Z[:-1])[:, :, -1] ** 2).mean(axis=(0, 1)) > 1e-4
    names = [n for n, ok in zip(meta["body"], live) if ok]
    Z = Z[:, :, live]
    D = deltas(Z)
    T = D.shape[1]
    # each channel's yardstick is its own 'nothing changes' error, floored at 1% of a typical channel's, so a channel
    # that hardly moves cannot make a tiny miss look huge
    ms = (D[:-1] ** 2).mean(axis=(0, 1))                    # [horizons, channels], from training lives
    scale = torch.tensor(np.maximum(ms, 0.01 * np.median(ms, axis=1, keepdims=True)))
    X = torch.tensor(np.concatenate([Z, world], axis=2)[:, :T])
    Y = torch.tensor(D)
    Xtr, Ytr, Xte, Yte = X[:-1], Y[:-1], X[-1:], Y[-1:]
    # 'same as yesterday': the change ahead = the change at this minute one day earlier
    yday = torch.cat([torch.zeros_like(Y[:, :DAY]), Y[:, :-DAY]], dim=1)

    # the held-out life, cut into days that each start with 6 hours of warm-up so his state is settled
    starts = list(range(CHUNK, T - DAY, max(DAY, (T - DAY - CHUNK) // 20)))[:20]
    ex = torch.cat([Xte[:, s - CHUNK:s + DAY] for s in starts])
    ey = torch.cat([Yte[:, s:s + DAY] for s in starts])
    eyd = torch.cat([yday[-1:, s:s + DAY] for s in starts])
    ok = torch.tensor([s >= DAY for s in starts])

    brain = Brain(X.shape[2], len(names), hidden)
    opt = torch.optim.Adam(brain.parameters(), lr=1e-3)
    params = sum(p.numel() for p in brain.parameters())
    print(f"{len(names)} body channels ({int((~live).sum())} never move), {world.shape[2]} world channels, "
          f"{Xtr.shape[0]} lives x {T / DAY:.1f} days to learn from, 1 life held out; brain {params:,} numbers",
          flush=True)

    def held_out():
        with torch.no_grad():
            p, _ = brain(ex)
            p = p[:, CHUNK:]
            s = surprise(p[ok], ey[ok], scale)
            return dict(model=s.tolist(), yesterday=surprise(eyd[ok], ey[ok], scale).tolist(),
                        per_channel=(((p[ok] - ey[ok]) ** 2 / scale).mean(dim=(0, 1))).tolist())

    curve, checks = [], []
    h, t0, step = None, time.time(), 0
    for c0 in range(0, T - CHUNK + 1, CHUNK):
        x, y = Xtr[:, c0:c0 + CHUNK], Ytr[:, c0:c0 + CHUNK]
        pred, hn = brain(x, h)
        s = surprise(pred, y, scale)                        # what this stretch surprised him by, before learning
        row = dict(days=(c0 + CHUNK) / DAY, model=s.tolist())
        if c0 >= DAY:
            row["yesterday"] = surprise(yday[:-1, c0:c0 + CHUNK], y, scale).tolist()
        curve.append(row)
        opt.zero_grad()
        s.mean().backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        h = hn.detach()                                     # his state carries on; only the learning is cut here
        step += 1
        if step % 20 == 0 or c0 + 2 * CHUNK > T:
            checks.append(dict(days=row["days"], seconds=round(time.time() - t0, 1), **held_out()))
            c = checks[-1]
            print(f"day {row['days']:6.1f}  held-out skill 1/15/60 min: "
                  + " / ".join(f"{1 - v:.3f}" for v in c["model"])
                  + "   'same as yesterday': " + " / ".join(f"{1 - v:.3f}" for v in c["yesterday"]), flush=True)
        if time.time() - t0 > seconds:
            print("time is up", flush=True)
            break
    if not checks or checks[-1]["days"] != curve[-1]["days"]:
        checks.append(dict(days=curve[-1]["days"], seconds=round(time.time() - t0, 1), **held_out()))
    torch.save(brain.state_dict(), os.path.join(out, "brain.pt"))
    json.dump(dict(src=src, hidden=hidden, params=params, horizons=H, channels=names, lives=int(Xtr.shape[0]),
                   curve=curve, checks=checks), open(os.path.join(out, "learned.json"), "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(H), figsize=(13, 3.8), sharey=True)
    for i, (ax, hz) in enumerate(zip(axes, H)):
        days = [r["days"] for r in curve]
        live_s = np.array([r["model"][i] for r in curve])
        ax.plot(days, live_s, color="#999", lw=0.6, label="while living (before learning each stretch)")
        ax.plot([c["days"] for c in checks], [c["model"][i] for c in checks], "o-", color="#c33", ms=3,
                label="held-out life")
        ax.axhline(checks[-1]["yesterday"][i], color="#36c", ls="--", lw=1, label="'same as yesterday'")
        ax.axhline(1.0, color="k", ls=":", lw=1, label="'nothing changes'")
        ax.set_title(f"{hz} min ahead")
        ax.set_xlabel("days lived (per life)")
        ax.set_yscale("log")
    axes[0].set_ylabel("surprise (1 = 'nothing changes')")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Life model v0: his surprise as he lives ({Xtr.shape[0]} lives, brain {params:,} numbers)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "surprise.png"), dpi=130)
    print("saved", out, flush=True)
