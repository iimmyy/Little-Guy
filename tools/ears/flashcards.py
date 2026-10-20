#!/usr/bin/env python3
"""Sound flashcards for widening his ears to 18 kHz: short clips cut from a sample library, each with a name taken
from its file name (no AI labels anywhere: the file names are the answers, and the brightness answers are measured).

  pick  LIST.txt ROOT OUT.jsonl [--per 500] [--held 0.1]
        LIST.txt = file paths relative to ROOT (one per line, from `find`). A clip's name comes from the first rule
        below that matches its file name, then its folder names. Whole packs (the first two folders) are held out
        for the test, so the test sounds come from packs he never trained on. Anything with "splice" in its path is
        left out (Splice's license forbids AI training of any kind).
  extra OUT.jsonl NAME ROOT LIST.txt [--crops 3] [--held 0.1] [--skip WORD ...]
        adds whole recordings under one name ("a singing voice", "a song"), a few crops each.
  cut   PICKS.jsonl OUT_DIR [--seconds 4] [--jobs 8]
        decodes each pick to 48 kHz mono, cuts up to SECONDS (the loudest stretch for long files), writes 16-bit FLAC
        and adds "clip" (file in OUT_DIR) and "reach_khz" (the highest 500 Hz band within 70 dB of the 200-4000 Hz
        level, before any training-time filtering).
"""
import hashlib, json, os, random, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# first match wins; the order puts specific names before general ones (an "open hat" before a "hat")
RULES = [
    ("an open hi-hat", r"open.?h(i.?)?hat|\boh\b|open.?hh"),
    ("a closed hi-hat", r"closed.?h(i.?)?hat|\bch\b|closed.?hh"),
    ("a hi-hat", r"hi.?hat|\bhh\b|\bhats?\b"),
    ("a crash cymbal", r"crash"),
    ("a ride cymbal", r"\bride"),
    ("a cymbal", r"cymbal|\bsplash\b|\bchina\b"),
    ("a shaker", r"shaker|shake\b"),
    ("a tambourine", r"tamb"),
    ("a clap", r"clap"),
    ("a snap", r"snap"),
    ("a rimshot", r"\brim"),
    ("a snare", r"snare"),
    ("an 808", r"808"),
    ("a kick drum", r"kick|\bbd\b"),
    ("a tom", r"\btoms?\b"),
    ("a conga or bongo", r"conga|bongo"),
    ("a cowbell", r"cowbell"),
    ("a riser", r"riser|uplifter|build.?up"),
    ("a downlifter or sweep", r"downlifter|sweep|whoosh"),
    ("an impact", r"impact|\bhit\b"),
    ("a vocal chop", r"vocal|vox|chant"),
    ("a bass", r"\bbass"),
    ("a piano", r"piano|\bkeys\b|rhodes"),
    ("a guitar", r"guitar"),
    ("a bell", r"\bbells?\b|glock|chime"),
    ("a pluck", r"pluck"),
    ("a pad", r"\bpads?\b"),
    ("strings", r"string|violin|cello"),
    ("a flute", r"flute"),
    ("brass", r"brass|trumpet|horn|sax"),
    ("a synth lead", r"\blead\b|synth"),
]
AUDIO = (".wav", ".aif", ".aiff", ".flac")


def name_of(path):
    low = path.lower().replace("_", " ").replace("-", " ")
    base = os.path.basename(low)
    for where in (base, low):                   # the file name decides first, then its folders
        for name, pat in RULES:
            if re.search(pat, where):
                return name
    return None


def pack_of(path):
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def held_packs(packs, frac, seed=0):
    packs = sorted(set(packs))
    random.Random(seed).shuffle(packs)
    return set(packs[: max(1, round(len(packs) * frac))])


def pick(lst, root, out, per=500, held=0.1):
    per, held = int(per), float(held)
    files = [l.strip().lstrip("./") for l in open(lst) if l.strip().lower().endswith(AUDIO)]
    files = [f for f in files if "splice" not in f.lower()]
    test = held_packs([pack_of(f) for f in files], held)
    by = {}
    for f in files:
        n = name_of(f)
        if n and "loop" not in f.lower():       # single hits and short sounds only: a loop is several things at once
            by.setdefault(n, []).append(f)
    rng = random.Random(1)
    with open(out, "w") as o:
        for n, fs in sorted(by.items()):
            # spread across packs: take one per pack in turn until the cap, so one big pack can't fill a name
            packs = {}
            for f in fs:
                packs.setdefault(pack_of(f), []).append(f)
            for v in packs.values():
                rng.shuffle(v)
            order, keys = [], list(packs)
            rng.shuffle(keys)
            while len(order) < per and any(packs.values()):
                for k in keys:
                    if packs[k] and len(order) < per:
                        order.append(packs[k].pop())
            for f in order:
                o.write(json.dumps({"src": os.path.join(root, f), "name": n, "pack": pack_of(f),
                                    "split": "test" if pack_of(f) in test else "train"}) + "\n")
            print(f"{n:24s} {len(order):4d} of {len(fs)}")


def extra(out, name, root, lst, *rest):
    crops, held, skip = 3, 0.1, []
    i = 0
    while i < len(rest):
        if rest[i] == "--crops": crops = int(rest[i + 1]); i += 2
        elif rest[i] == "--held": held = float(rest[i + 1]); i += 2
        elif rest[i] == "--skip":
            i += 1
            while i < len(rest) and not rest[i].startswith("--"):
                skip.append(rest[i].lower()); i += 1
        else: raise SystemExit("unknown " + rest[i])
    files = [l.strip().lstrip("./") for l in open(lst) if l.strip()]
    files = [f for f in files if "splice" not in f.lower() and not any(s in f.lower() for s in skip)]
    test = held_packs(files, held, seed=2)      # each recording is its own "pack"
    with open(out, "a") as o:
        for f in files:
            for c in range(crops):
                o.write(json.dumps({"src": os.path.join(root, f), "name": name, "pack": f, "crop": c,
                                    "split": "test" if f in test else "train"}) + "\n")
    print(name, len(files), "recordings x", crops, "crops,", sum(f in test for f in files), "held out")


def reach_khz(x, sr=48000):
    n = 4096
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    fr = x[: len(x) // n * n].reshape(-1, n) * np.hanning(n)
    P = (np.abs(np.fft.rfft(fr, axis=1)) ** 2).mean(0)
    hz = np.fft.rfftfreq(n, 1 / sr)
    band = lambda a, b: P[(hz >= a) & (hz < b)].mean()
    ref = band(200, 4000)
    top = max([a for a in range(0, 23500, 500) if band(a, a + 500) > ref * 1e-7] or [0])
    return (top + 500) / 1000


def cut_one(row, out_dir, seconds):
    key = hashlib.md5(f"{row['src']}|{row.get('crop', 0)}".encode()).hexdigest()[:16]
    dst = os.path.join(out_dir, key + ".flac")
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", row["src"], "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                             capture_output=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        return None
    x = np.frombuffer(raw, np.float32)
    if len(x) < 0.1 * 48000 or not np.isfinite(x).all() or np.abs(x).max() < 1e-4:
        return None
    L = int(seconds * 48000)
    if len(x) > L:
        if "crop" in row:                       # whole recordings: evenly spaced crops
            starts = np.linspace(0, len(x) - L, 5)[1:-1] if len(x) > 3 * L else np.linspace(0, len(x) - L, 3)
            s = int(starts[row["crop"] % len(starts)])
        else:                                   # a sound: its loudest stretch
            e = np.convolve(x ** 2, np.ones(4800), "valid")[::480]
            s = max(0, min(len(x) - L, int(np.argmax(e) * 480) - L // 4))
        x = x[s: s + L]
    x = x / max(1.0, float(np.abs(x).max()))
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", "48000", "-ac", "1", "-i", "-",
                    "-compression_level", "5", dst], input=pcm, check=True)
    return {**row, "clip": os.path.basename(dst), "seconds": round(len(x) / 48000, 3), "reach_khz": reach_khz(x)}


def cut(picks, out_dir, *rest):
    seconds, jobs = 4.0, 8
    for i in range(0, len(rest), 2):
        if rest[i] == "--seconds": seconds = float(rest[i + 1])
        elif rest[i] == "--jobs": jobs = int(rest[i + 1])
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(picks)]
    done = 0
    with ThreadPoolExecutor(jobs) as ex, open(os.path.join(out_dir, "cards.jsonl"), "x") as o:
        for r in ex.map(lambda r: cut_one(r, out_dir, seconds), rows):
            if r:
                o.write(json.dumps(r) + "\n"); done += 1
    print(f"cut {done} of {len(rows)} into {out_dir}")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "pick":
        pos = [a for a in args if not a.startswith("--")]
        kw = {args[i][2:]: args[i + 1] for i in range(len(args)) if args[i].startswith("--")}
        pos = [a for a in pos if a not in kw.values()]
        pick(*pos, **kw)
    elif cmd == "extra":
        extra(*args)
    elif cmd == "cut":
        cut(*args)
    else:
        raise SystemExit(__doc__)
