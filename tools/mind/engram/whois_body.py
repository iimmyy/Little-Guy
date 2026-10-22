"""Who's talking, with his body on (Immy, 2026-10-22: "lets rerun that text test with this engine").

The same asks as whois.py (Immy's messages vs Claude's, with three examples of each and cold), sent through his
brainstem's live choice path (his face first, then the odds of each name) twice with the same seed: once with no drip
(body off) and once with the drip his body gives after hearing that message (body on). The body is his own, copied
from his room (chem.json) and his zero (state.json); each message gets a fresh copy of it, then the same appraisal his
live mind makes before answering (tools/room/mind_live.py answer()): how the words read to him (threat if they read
below his zero), surprise and novelty, a reward event, one minute of company. Messages are told to him as coming in, so
nobody is counted as here before them.

  python whois_body.py --ask ASK.jsonl --chem chem.json --state state.json --key KEY --port 8781 --out DIR
  -> DIR/rows.jsonl, DIR/grade.txt
"""
import argparse, copy, json, math, os, sys, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chem"))
import engine as chem

# the same numbers as mind_live.py (kept in step by hand: that file can't be imported outside his room)
DIAL_V = {"wanting": 0.4, "liking": 0.4, "comfort": 0.6, "warmth": 0.6, "dysphoria": -0.6, "anxious": -0.3, "sick": -0.6}
DIAL_A = {"awake": 1.0, "anxious": 0.5, "sympathetic": 0.3}
THREAT_FROM, THREAT_FULL, CEIL, CEIL_AROUSAL = 1.0, 3.0, 0.75, 0.5
TURN, END = "<bos><|turn>user\n", "<turn|>\n<|turn>model\n"


def soft(x, ceil):
    return ceil * math.tanh(x / ceil)


def call(args, path, **req):   # req carries v and a (his dials), so the run's settings are args here
    r = urllib.request.Request(f"http://127.0.0.1:{args.port}/{path}", data=json.dumps(req).encode(),
                               headers={"X-LG-Key": args.key_text, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.loads(f.read())


def heard(a, body, rest, zero, text):
    """his body hearing one message: the appraisal, one minute, then the dials his brain gets"""
    e = copy.deepcopy(body)
    raw = call(a, "read", text=text[:8000])["feel"]
    rd = [raw[0] - zero[0][0], raw[1] - zero[0][1]]                  # against how incoming messages usually read
    threat = min(1.0, max(0.0, (-rd[0] / zero[1][0] - THREAT_FROM) / (THREAT_FULL - THREAT_FROM)))
    rw = e.reward_event("message", 1.0 - threat)
    e.step(1 / 60, awake=True, surprise=0.1, novelty=0.0, threat=threat, company=1.0)
    k = e.knobs()
    r = lambda n: (k[n] - rest[n]) / max(abs(rest[n]), 0.1)
    bv, ba = sum(w * r(n) for n, w in DIAL_V.items()), sum(w * r(n) for n, w in DIAL_A.items())
    v = soft(bv + 0.3 * rw["delta"], CEIL)
    ar = soft(ba + 0.2 * max(0.0, rw["delta"]), CEIL_AROUSAL)
    return dict(v=v, a=ar, read=rd, threat=threat, da_delta=rw["delta"], ne=e.c["ne"], crh=e.c["crh"])


def run(a):
    a.key_text = open(a.key).read().strip()
    body = object.__new__(chem.Chem)                       # his body as plain data from his room, never code
    body.__dict__.update(json.load(open(a.chem)))
    chem.CAL = body.cal
    st = json.load(open(a.state))
    rest, zero = st["chem_rest"], (st["zero"]["heard"], st["zero"]["heard_sd"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    asks = [json.loads(l) for l in open(a.ask)]
    felt = {}
    with open(out / "rows.jsonl", "w") as f:
        for i, q in enumerate(asks):
            text = q["parts"][0]["text"]
            assert text.startswith(TURN) and text.endswith(END), "ask format changed"
            msgs = [{"role": "user", "content": text[len(TURN):-len(END)]}]
            msg = q["meta"]["text"]
            if msg not in felt:
                felt[msg] = heard(a, body, rest, zero, msg)
            b = felt[msg]
            row = {"meta": q["meta"], "body": {k: round(v, 4) if isinstance(v, float) else v for k, v in b.items()}}
            for cond, (v, ar) in (("off", (0.0, 0.0)), ("on", (b["v"], b["a"]))):
                r = call(a, "choose", messages=msgs, choices=q["cands"], v=v, a=ar, seed=1000 + i)
                lp = r["logprobs"]
                row[cond] = {"said": max(lp, key=lp.get), "logprobs": lp, "face": r.get("face", "")}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 20 == 0:
                print(i, len(asks), msg[:50], row["off"]["said"], row["on"]["said"], round(b["v"], 3), round(b["a"], 3),
                      flush=True)


def grade(a):
    rows = [json.loads(l) for l in open(Path(a.out) / "rows.jsonl")]
    lines = []
    name = {"immy": "Immy", "claude": "Claude"}
    for s in sorted({r["meta"]["set"] for r in rows}):
        for way in sorted({r["meta"]["way"] for r in rows}):
            rs = [r for r in rows if r["meta"]["set"] == s and r["meta"]["way"] == way]
            if not rs:
                continue
            parts = []
            for cond in ("off", "on"):
                ok = [r for r in rs if r[cond]["said"] == name[r["meta"]["who"]]]
                im = sum(r["meta"]["who"] == "immy" for r in ok)
                parts.append(f"body {cond}: right {len(ok)}/{len(rs)} (Immy's {im}, Claude's {len(ok) - im})")
            flips = sum(r["off"]["said"] != r["on"]["said"] for r in rs)
            lines.append(f"{s} {way}: " + "; ".join(parts) + f"; answers changed {flips}")
    vs = [r["body"]["v"] for r in rows]
    ars = [r["body"]["a"] for r in rows]
    th = [r["body"]["threat"] for r in rows]
    lines.append(f"his dials with the body on: valence {min(vs):+.3f} to {max(vs):+.3f} (mean {sum(vs) / len(vs):+.3f}), "
                 f"arousal {min(ars):+.3f} to {max(ars):+.3f}; messages read as a threat {sum(t > 0 for t in th)}/{len(th)}")
    txt = "\n".join(lines)
    print(txt)
    (Path(a.out) / "grade.txt").write_text(txt + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "grade"])
    p.add_argument("--ask"); p.add_argument("--chem"); p.add_argument("--state"); p.add_argument("--key")
    p.add_argument("--port", type=int, default=8781); p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a) if a.cmd == "run" else None
    grade(a)
