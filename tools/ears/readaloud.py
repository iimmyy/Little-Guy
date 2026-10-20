#!/usr/bin/env python3
"""Read-aloud word lessons for his ears. Immy reads a script in one take with a pause between lines, then exports
copies of the same take through effects. Every copy is lined up with the dry take and cut at the dry take's places,
so each line comes out once per version: the script line is the exact answer, the effect is its difficulty.

  cut SCRIPT.tsv TAKES_DIR OUT_DIR [--redo 12,30]
      SCRIPT.tsv = line, kind, pair, text (tools/ears/read-aloud/script-NN.tsv).
      TAKES_DIR holds sNN_dry.wav and sNN_<effect>-<amount>[+<effect>-<amount>...].wav (any format ffmpeg reads).
      --redo = lines read twice (a flub, then the line again); the first try is dropped.
      Writes OUT_DIR/clips/sNN-LNN-<version>.flac (48 kHz mono 16-bit), OUT_DIR/lessons.jsonl (one row per clip:
      audio, dry, text, line, kind, pair, version, effects, seconds) and OUT_DIR/cuts.json (what was found, and
      every warning, for reading before training).

Lines are found by silence: with N readings, the N-1 longest pauses inside the take are the breaks between lines.
That holds as long as Immy's pauses between lines are longer than any pause inside a line; cuts.json reports the
margin (shortest break taken vs longest pause left inside a line) so a take that breaks this shows up.
Every version of a line gets the same window: from just before the words start to the middle of the next break
(or 1 s after the words end, whichever comes first), so reverb and delay tails stay with their line.
"""
import json, os, re, subprocess, sys

import numpy as np

SR = 48000
HOP = 480                      # 10 ms frames for finding speech
LEAD, TAIL = 0.15, 1.0         # seconds kept before the words, most kept after


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def save(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-c:a", "flac", path], input=pcm, check=True)


def frames_db(x):
    n = len(x) // HOP
    rms = np.sqrt(np.mean(x[:n * HOP].reshape(n, HOP) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def pauses(speech):
    """Runs of non-speech frames strictly inside the take, as (start, end) frame pairs."""
    idx = np.flatnonzero(speech)
    first, last = idx[0], idx[-1]
    runs, i = [], first
    while i <= last:
        if not speech[i]:
            j = i
            while not speech[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return first, last + 1, runs


def find_lines(x, n):
    db = frames_db(x)
    floor = np.percentile(db, 10)
    thresh = max(floor + 15, db.max() - 50)
    first, end, runs = pauses(db > thresh)
    if len(runs) < n - 1:
        raise SystemExit(f"found {len(runs) + 1} stretches of speech, expected {n}: are the pauses there?")
    by_len = sorted(runs, key=lambda r: r[1] - r[0], reverse=True)
    breaks = sorted(by_len[:n - 1])
    shortest_break = min(b - a for a, b in breaks) * HOP / SR if breaks else None
    longest_inside = (by_len[n - 1][1] - by_len[n - 1][0]) * HOP / SR if len(by_len) >= n else 0.0
    starts = [first] + [b for _, b in breaks]
    ends = [a for a, _ in breaks] + [end]
    mids = [(a + b) // 2 for a, b in breaks]
    lines = []
    for k in range(n):
        s, e = starts[k] * HOP / SR, ends[k] * HOP / SR
        lo = max(0.0, s - LEAD) if k == 0 else max(mids[k - 1] * HOP / SR, s - LEAD)
        hi = e + TAIL if k == n - 1 else min(mids[k] * HOP / SR, e + TAIL)
        lines.append({"words_from": round(s, 3), "words_to": round(e, 3), "from": round(lo, 3), "to": round(hi, 3)})
    return lines, {"threshold_db": round(float(thresh), 1), "floor_db": round(float(floor), 1),
                   "shortest_break_s": shortest_break, "longest_pause_inside_s": longest_inside}


def lag_of(dry, wet, most=2.0):
    """Samples to shift wet by so it lines up with dry, from 1 ms loudness envelopes (good to about 1 ms)."""
    b = SR // 1000

    def env(x):
        n = len(x) // b
        e = np.abs(x[:n * b]).reshape(n, b).mean(axis=1)
        return (e - e.mean()) / (e.std() + 1e-9)
    a, w = env(dry), env(wet)
    m = int(most * 1000)
    size = 1 << int(np.ceil(np.log2(len(a) + len(w))))
    c = np.fft.irfft(np.fft.rfft(w, size) * np.conj(np.fft.rfft(a, size)), size)
    c = np.concatenate([c[-m:], c[:m + 1]])            # lags -m .. +m
    k = int(np.argmax(c))
    return (k - m) * b, float(c[k] / min(len(a), len(w)))


def parse_version(name):
    effects = []
    for part in name.split("+"):
        eff, _, amount = part.rpartition("-") if "-" in part else (part, "", "")
        effects.append({"effect": eff.lower(), "amount": amount.lower()})
    return effects


def cut(script, takes, out, *rest):
    redo = set()
    if "--redo" in rest:
        redo = {int(v) for v in rest[rest.index("--redo") + 1].split(",") if v.strip()}
    rows = [dict(zip(["line", "kind", "pair", "text"], l.rstrip("\n").split("\t")))
            for l in open(script).readlines()[1:] if l.strip()]
    files = sorted(f for f in os.listdir(takes) if not f.startswith("."))
    dry_name = next((f for f in files if re.match(r"s\d+_dry\.\w+$", f)), None)
    if not dry_name:
        raise SystemExit("no sNN_dry file in " + takes)
    session = dry_name.split("_")[0]
    dry = load(os.path.join(takes, dry_name))
    readings = []
    for r in rows:
        if int(r["line"]) in redo:
            readings.append(None)                       # the flubbed first try, dropped
        readings.append(r)
    spans, found = find_lines(dry, len(readings))
    warnings = []
    if found["shortest_break_s"] is not None and found["shortest_break_s"] < 1.5 * found["longest_pause_inside_s"]:
        warnings.append(f"breaks between lines ({found['shortest_break_s']:.2f} s) are not clearly longer than pauses "
                        f"inside lines ({found['longest_pause_inside_s']:.2f} s): check the cuts by listening")
    os.makedirs(os.path.join(out, "clips"), exist_ok=True)
    versions = {"dry": (dry, 0, 1.0)}
    for f in files:
        m = re.match(rf"{session}_(.+)\.\w+$", f)
        if not m or f == dry_name:
            continue
        wet = load(os.path.join(takes, f))
        lag, match = lag_of(dry, wet)
        wet = np.concatenate([np.zeros(-lag, np.float32), wet]) if lag < 0 else wet[lag:]
        if abs(lag) > SR:
            warnings.append(f"{f}: started {lag / SR:+.2f} s off the dry take (lined up, but check it)")
        if match < 0.3:
            warnings.append(f"{f}: lines up poorly with the dry take (match {match:.2f}); was its timing changed?")
        versions[m.group(1)] = (wet, lag, match)
    lessons, cuts = [], {"session": session, "found": found, "lines": [], "versions": {}, "warnings": warnings}
    for name, (x, lag, match) in versions.items():
        peak = float(np.abs(x).max())
        gain = 0.99 / peak if peak > 0.99 else 1.0      # one gain per version, so its lines keep their balance
        cuts["versions"][name] = {"lag_ms": round(lag / SR * 1000, 1), "match": round(match, 3), "gain": round(gain, 3),
                                  "effects": [] if name == "dry" else parse_version(name)}
        for r, sp in zip(readings, spans):
            if r is None:
                continue
            clip = x[int(sp["from"] * SR):int(sp["to"] * SR)] * gain
            path = os.path.join("clips", f"{session}-L{int(r['line']):02d}-{name}.flac")
            save(os.path.join(out, path), clip)
            lessons.append({"audio": path, "dry": os.path.join("clips", f"{session}-L{int(r['line']):02d}-dry.flac"),
                            "text": r["text"], "line": int(r["line"]), "kind": r["kind"], "pair": r["pair"],
                            "session": session, "version": name, "effects": cuts["versions"][name]["effects"],
                            "seconds": round(len(clip) / SR, 2)})
    for r, sp in zip(readings, spans):
        words = len(r["text"].split()) if r else None
        secs = sp["words_to"] - sp["words_from"]
        row = dict(sp, line=int(r["line"]) if r else None, dropped=r is None)
        if r and not 0.8 <= words / max(secs, 1e-3) <= 6.0:
            row["check"] = f"{words} words in {secs:.1f} s"
            warnings.append(f"line {r['line']}: {words} words in {secs:.1f} s, so the cuts may be off by a line here")
        cuts["lines"].append(row)
    with open(os.path.join(out, "lessons.jsonl"), "w") as f:
        for l in lessons:
            f.write(json.dumps(l) + "\n")
    json.dump(cuts, open(os.path.join(out, "cuts.json"), "w"), indent=1)
    print(json.dumps({"session": session, "lines": len(rows), "versions": list(versions), "clips": len(lessons),
                      "found": found, "warnings": warnings}, indent=1))


if __name__ == "__main__":
    {"cut": cut}[sys.argv[1]](*sys.argv[2:])
