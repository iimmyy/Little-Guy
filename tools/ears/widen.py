#!/usr/bin/env python3
"""Widening his ears from 8 to 18 kHz (Gemma 4 E4B's audio tower), without changing what he already hears.

His ears take 16 kHz sound as 128 bands (0-8 kHz) through two small convolutions and a projection
(subsample_conv_projection). This adds a side path, like a LoRA on the input:
  - his old path is untouched: it hears the sound resampled to 16 kHz, exactly as before
  - a new front end takes the 48 kHz sound in the same 10 ms steps and makes 39 more bands, 8 to 18 kHz, spaced the
    way his 128 are (HTK mel, the same width per band; the first starts where his top band ends), measured from the
    floor
  - a copy of his two convolutions reads those bands, and a projection that starts at zero adds the result to his
    old path's output, so on day one he hears exactly what he heard
Only the side path trains; his ears' other layers and his brain stay frozen.
Sound with nothing above 8 kHz is NOT silent in the new bands: loud sound just below 8 kHz smears across the line in
any 20 ms window (measured: a 16 kHz round trip still reaches a median 6.6 there, the untouched clip 8.3). So dull
versions are part of the lessons ("up to about 8 kHz", and naming them), and `check` measures that naming dull sound
doesn't get worse.

What it learns from (the answers are measured or come from file names, never from an AI):
  name   "What is this sound?"                -> the card's name ("a closed hi-hat", "a singing voice", "a song")
  reach  "Up to what frequency does it go?"    -> measured after a random cut (a brick wall at 9-17 kHz, the 16 kHz
                                                   round trip, or none): the highest band within 50 dB of the loudest
  air    "How much high end does it have?"     -> measured share of energy above 8 kHz, in four steps

  train  CARDS_DIR MODEL OUT_DIR [--steps 6000] [--lr 3e-4] [--accum 8]
  eval   CARDS_DIR MODEL OUT.json [--side OUT_DIR/side.pt] [--n 300]   held-out packs; without --side = the old ears
         (name_right_dull: naming the 16 kHz round trip, the check that the side path doesn't hurt old hearing)
  listen MODEL WAV OUT.json [--side ...] [--ask "..."]...               free answers to a full-range recording
"""
import json, math, os, random, re, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

SR = 48000
N_OLD, N_NEW = 128, 39
FLOOR = 1e-3


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def high_filters(n_fft=1024, sr=SR):
    """39 triangles continuing his 128 at the same spacing (mel(8 kHz)/129): the first starts exactly where his top band
    ends (8 kHz), so the two never share a frequency; the last ends just above 18 kHz"""
    d = hz_to_mel(8000.0) / (N_OLD + 1)
    centers = hz_to_mel(8000.0) + d * np.arange(0, N_NEW + 2)           # edges: lower, center..., upper
    edges = mel_to_hz(centers)
    fr = np.fft.rfftfreq(n_fft, 1 / sr)
    W = np.zeros((len(fr), N_NEW), np.float32)
    for j in range(N_NEW):
        lo, c, hi = edges[j], edges[j + 1], edges[j + 2]
        up = (fr - lo) / (c - lo)
        down = (hi - fr) / (hi - c)
        W[:, j] = np.maximum(0, np.minimum(up, down))
    return torch.from_numpy(W), edges


class HighBands(nn.Module):
    """48 kHz sound -> [T, 39] log bands above 8 kHz, minus the floor (silence = 0), on his ears' clock: frame i covers
    the same 20 ms his frame i does (semicausal: 10 ms of zeros in front, frames of 20 ms every 10 ms)"""
    def __init__(self):
        super().__init__()
        W, self.edges = high_filters()
        self.register_buffer("W", W)
        self.register_buffer("win", torch.hann_window(960, periodic=True))

    def forward(self, x48, n_frames):
        x = F.pad(x48, (480, 0))
        need = (n_frames - 1) * 480 + 960
        if x.shape[-1] < need:
            x = F.pad(x, (0, need - x.shape[-1]))
        fr = x.unfold(-1, 960, 480)[:n_frames] * self.win
        mag = torch.fft.rfft(fr, n=1024).abs() / 3.0          # / 3: the same scale his 16 kHz frames have (3x fewer samples)
        return torch.log(mag @ self.W + FLOOR) - math.log(FLOOR)


class Side(nn.Module):
    """reads the high bands with a copy of his two convolutions; its projection starts at zero"""
    def __init__(self, proj, hidden):
        super().__init__()
        def layer(old):
            c = nn.Conv2d(old.conv.in_channels, old.conv.out_channels, 3, stride=2, padding=1, bias=False)
            c.weight.data.copy_(old.conv.weight.data.float())
            n = nn.LayerNorm(old.conv.out_channels, eps=old.norm.eps, elementwise_affine=True, bias=False)
            n.weight.data.copy_(old.norm.weight.data.float())
            return nn.ModuleList([c, n])
        self.l0, self.l1 = layer(proj.layer0), layer(proj.layer1)
        n_freq = math.ceil(math.ceil(N_NEW / 2) / 2)
        self.out = nn.Linear(n_freq * proj.layer1.conv.out_channels, hidden, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, high, mask=None):                       # high [B, T, 39]
        h = high.unsqueeze(1)
        for conv, norm in (self.l0, self.l1):
            if mask is not None:
                h = h * mask[:, None, :, None].to(h.dtype)
            h = F.relu(norm(conv(h).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            if mask is not None:
                mask = mask[:, ::2]
        B, _, T, _ = h.shape
        return self.out(h.permute(0, 2, 3, 1).reshape(B, T, -1))


def attach(model):
    """puts the side path on the model's ears; set model._high before each forward (None = old ears only)"""
    tower = model.model.audio_tower
    proj = tower.subsample_conv_projection
    side = Side(proj, tower.config.hidden_size).to(next(proj.parameters()).device)
    old = proj.forward

    def forward(input_features, input_features_mask=None):
        h, m = old(input_features, input_features_mask)
        hi = getattr(model, "_high", None)
        if hi is not None:
            add = side(hi.to(h.device, torch.float32), input_features_mask)
            h = h + add[:, : h.shape[1]].to(h.dtype)
        return h, m
    proj.forward = forward
    model._side = side
    return side


def load(model_dir):
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_dir)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return proc, model


def read_flac(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR, path
    return x


def lowpass(x, khz):
    """None = untouched; 8 = the 16 kHz round trip his old ears get (nothing real above 8 kHz); else a brick wall: every
    frequency above the cut is zeroed, so the sound really stops there"""
    if khz is None:
        return x
    if khz == 8:
        return resample_poly(resample_poly(x, 1, 3), 3, 1)[: len(x)].astype(np.float32)
    X = np.fft.rfft(x)
    X[np.fft.rfftfreq(len(x), 1 / SR) > khz * 1000] = 0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def reach_khz(x):
    """where the sound stops: the top of the highest 500 Hz band within 50 dB of the loudest one"""
    n = 4096
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    fr = x[: len(x) // n * n].reshape(-1, n) * np.hanning(n)
    P = (np.abs(np.fft.rfft(fr, axis=1)) ** 2).mean(0)
    hz = np.fft.rfftfreq(n, 1 / SR)
    bands = np.array([P[(hz >= a) & (hz < a + 500)].mean() for a in range(0, 24000, 500)])
    if bands.max() <= 0:
        return 0.0
    return (np.nonzero(bands > bands.max() * 1e-5)[0].max() + 1) * 0.5


def air_share(x):
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    hz = np.fft.rfftfreq(len(x), 1 / SR)
    return float(S[hz > 8000].sum() / max(S.sum(), 1e-20))


AIR = [(0.003, "almost none"), (0.03, "a little"), (0.15, "a fair amount"), (9e9, "a lot")]
Q = {"name": "What is this sound? Just name it in a few words.",
     "reach": "Up to roughly what frequency does this sound go? Answer like: Up to about N kHz.",
     "air": "How much high end does this have: almost none, a little, a fair amount, or a lot?"}


def card_example(card, x, rng):
    """one question about one clip; the clip may be lowpassed first (reach and air are measured after that)"""
    if np.abs(x).max() < 1e-3:
        return None                                           # a silent crop
    kind = rng.choice(["name", "reach", "reach", "air"])
    cut = rng.choice([None, None, None, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]) if kind != "name" else \
        rng.choice([None, None, 8])
    y = lowpass(x, cut)
    if kind == "name":
        ans = card["name"][0].upper() + card["name"][1:] + "."
    elif kind == "reach":
        r = reach_khz(y)
        if r < 7:
            return None                                       # stops well below the new bands: nothing to learn there
        r = min(r, cut) if cut else r                         # a cut's soft edge (the round trip) or window spill
        ans = "All the way up, past 18 kHz." if r > 18 else f"Up to about {max(8, round(r))} kHz."
    else:
        if reach_khz(y) < 7:
            return None
        a = air_share(y)
        ans = next(w for t, w in AIR if a < t)
        ans = ans[0].upper() + ans[1:] + "."
    return y, Q[kind], ans, kind


def encode(proc, model, high_fe, y48, question, answer=None):
    """the chat (sound, then the question), the old ears' features from the 16 kHz version, the new bands from the
    48 kHz one; with an answer: labels on the answer's tokens only"""
    y16 = resample_poly(y48, 1, 3).astype(np.float32)
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": y16}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + (answer + "<turn|>" if answer is not None else "")
    enc = proc(text=[full], audio=[y16], return_tensors="pt")
    dev = model.device
    enc = {k: v.to(dev) for k, v in enc.items()}
    T = enc["input_features"].shape[1]
    hi = high_fe(torch.from_numpy(y48).to(dev), T).unsqueeze(0)
    m = enc["input_features_mask"]
    hi = hi * m[..., None].to(hi.dtype)
    labels = None
    if answer is not None:
        n_prompt = proc(text=[prompt], audio=[y16], return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = -100
    return enc, hi, labels


def cards_of(cards_dir, split):
    rows = [json.loads(l) for l in open(os.path.join(cards_dir, "cards.jsonl"))]
    return [r for r in rows if r["split"] == split]


def train(cards_dir, model_dir, out_dir, *rest):
    kw = {rest[i][2:]: rest[i + 1] for i in range(0, len(rest), 2)}
    steps, lr, accum = int(kw.get("steps", 6000)), float(kw.get("lr", 3e-4)), int(kw.get("accum", 8))
    os.makedirs(out_dir, exist_ok=True)
    proc, model = load(model_dir)
    side = attach(model)
    high_fe = HighBands().to(model.device)
    opt = torch.optim.AdamW(side.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    cards = cards_of(cards_dir, "train")
    rng = random.Random(0)
    log = open(os.path.join(out_dir, "train.jsonl"), "a")
    seen, run = 0, {}
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        done = 0
        while done < accum:
            c = rng.choice(cards)
            ex = card_example(c, read_flac(os.path.join(cards_dir, "clips", c["clip"])), rng)
            if ex is None:
                continue
            y, q, a, kind = ex
            enc, hi, labels = encode(proc, model, high_fe, y, q, a)
            model._high = hi
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc, labels=labels).loss
            (loss / accum).backward()
            run.setdefault(kind, []).append(loss.item())
            done += 1; seen += 1
        torch.nn.utils.clip_grad_norm_(side.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 50 == 0:
            row = {"step": step + 1, "seen": seen, "lr": sched.get_last_lr()[0],
                   **{k: round(float(np.mean(v)), 4) for k, v in run.items()}}
            log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
            run = {}
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            ck = {"side": side.state_dict(), "edges_hz": high_fe.edges.tolist(), "step": step + 1}
            torch.save(ck, os.path.join(out_dir, f"side-{step + 1}.pt"))
            torch.save(ck, os.path.join(out_dir, "side.pt"))


@torch.no_grad()
def answer(proc, model, high_fe, y48, question, use_side, max_new=40):
    enc, hi, _ = encode(proc, model, high_fe, y48, question)
    model._high = hi if use_side else None
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def eval_(cards_dir, model_dir, out, *rest):
    kw = {rest[i][2:]: rest[i + 1] for i in range(0, len(rest), 2)}
    n = int(kw.get("n", 300))
    proc, model = load(model_dir)
    side = attach(model)
    if kw.get("side"):
        side.load_state_dict(torch.load(kw["side"], map_location=model.device)["side"])
    high_fe = HighBands().to(model.device)
    cards = cards_of(cards_dir, "test")
    rng = random.Random(123)
    rng.shuffle(cards)
    rows = []
    for c in cards[:n]:
        x = read_flac(os.path.join(cards_dir, "clips", c["clip"]))
        cut = rng.choice([None, 8, 10, 12, 14, 16])
        y = lowpass(x, cut)
        true_reach, air = reach_khz(y), air_share(y)
        true_reach = min(true_reach, cut) if cut else true_reach
        row = {"clip": c["clip"], "name": c["name"], "cut": cut, "reach": true_reach,
               "air": next(w for t, w in AIR if air < t)}
        for k in ("name", "reach", "air"):
            row["ans_" + k] = answer(proc, model, high_fe, y, Q[k], True)
        rows.append(row)
    def num(s):
        m = re.search(r"(\d+(?:\.\d+)?)\s*k", s or "")
        return float(m.group(1)) if m else None
    hi_rows = [r for r in rows if r["reach"] >= 7]
    def said(s):                                              # "past 18 kHz" counts as 18
        return 18.0 if s and "past 18" in s else num(s)
    errs = [abs(min(said(r["ans_reach"]), 18) - min(r["reach"], 18)) for r in hi_rows if said(r["ans_reach"]) is not None]
    summary = {
        "side": kw.get("side"), "n": len(rows),
        "name_right": round(np.mean([r["name"].split(" ", 1)[-1] in r["ans_name"].lower() for r in rows]), 3),
        "name_right_dull": round(np.mean([r["name"].split(" ", 1)[-1] in r["ans_name"].lower() for r in rows
                                          if r["cut"] == 8] or [float("nan")]), 3),
        "name_right_full": round(np.mean([r["name"].split(" ", 1)[-1] in r["ans_name"].lower() for r in rows
                                          if r["cut"] is None] or [float("nan")]), 3),
        "reach_answered": f"{len(errs)}/{len(hi_rows)}",
        "reach_mean_abs_err_khz": round(float(np.mean(errs)), 2) if errs else None,
        "reach_within_1khz": round(float(np.mean([e <= 1 for e in errs])), 3) if errs else None,
        "air_right": round(np.mean([r["air"] in r["ans_air"].lower() for r in hi_rows]), 3),
    }
    json.dump({"summary": summary, "rows": rows}, open(out, "x"), indent=1)
    print(json.dumps(summary))


def listen(model_dir, wav, out, *rest):
    side_pt, asks, i = None, [], 0
    while i < len(rest):
        if rest[i] == "--side": side_pt = rest[i + 1]
        elif rest[i] == "--ask": asks.append(rest[i + 1])
        i += 2
    asks = asks or ["What do you hear in this? Describe the sound, including the high end."]
    proc, model = load(model_dir)
    side = attach(model)
    if side_pt:
        side.load_state_dict(torch.load(side_pt, map_location=model.device)["side"])
    high_fe = HighBands().to(model.device)
    import soundfile as sf
    x, sr = sf.read(wav, dtype="float32", always_2d=False)
    x = x.mean(1) if x.ndim > 1 else x
    x = resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x
    x = x[: 30 * SR]
    res = [{"ask": q, "side": side_pt, "answer": answer(proc, model, high_fe, x, q, side_pt is not None, 120)} for q in asks]
    json.dump(res, open(out, "x"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"train": train, "eval": eval_, "listen": listen}[cmd](*args)
