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

  keep   "Transcribe the words" (several phrasings)  -> whatever the old ears answer, as a whole distribution: on
                                                   vocals and songs the side path must leave his words as they were
                                                   (output matching, a KL to the old ears; round 2 without it answered
                                                   "sing" when asked to transcribe a rap)

  keepset KEEP_DIR MODEL [--n 600]   the old ears' answers to transcription requests on KEEP_DIR/clips (cards.jsonl
         from flashcards.py cut), saved to KEEP_DIR/keep.jsonl; train --keep KEEP_DIR mixes them in (--keep_every 3:
         one keep lesson in every 3)
  train  CARDS_DIR MODEL OUT_DIR [--steps 6000] [--lr 3e-4] [--accum 8] [--keep KEEP_DIR --keep_every 3]
         [--where early|late|sep] [--mid 1024] [--pool 4]   (sep: the high end on its own tokens, see attach)
         [--init SIDE.pt] [--strict 1.0] [--balance 1]   init: start from a trained side path; strict: punish wrong
         guesses (strict_loss); balance: every reach/air answer equally common (card_example)
         [--talk ADAPTER_DIR]   train with his talk adapter on (his live personality layer, wear_talk); the keep
         lesson's teacher is still his brain without it, where he copies words faithfully
         [--listen RANK --listen_lr 1e-4]   round 8: his brain learns to listen too (wear_listen: a LoRA only on the
         audio positions he reads, never text), trained with the side path; saved in side.pt
  pagecheck MODEL WAV OUT.json   sep, day one: his old audio tokens and the brain's states over them are the same
         with and without the high-end page; what the page costs his answer before any training
  eval   CARDS_DIR MODEL OUT.json [--side OUT_DIR/side.pt] [--n 300] [--talk ADAPTER_DIR]   held-out packs; without --side = the old ears
         (name_right_dull: naming the 16 kHz round trip, the check that the side path doesn't hurt old hearing)
  listen MODEL WAV OUT.json [--side ...] [--talk ...] [--ask "..."]...  free answers to a full-range recording
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
    """reads the high bands with a copy of his two convolutions; its projection starts at zero. mid > 0 puts one
    hidden layer (GELU) before it."""
    def __init__(self, proj, hidden, mid=0, bias=False):
        super().__init__()
        def layer(old):
            c = nn.Conv2d(old.conv.in_channels, old.conv.out_channels, 3, stride=2, padding=1, bias=False)
            c.weight.data.copy_(old.conv.weight.data.float())
            n = nn.LayerNorm(old.conv.out_channels, eps=old.norm.eps, elementwise_affine=True, bias=False)
            n.weight.data.copy_(old.norm.weight.data.float())
            return nn.ModuleList([c, n])
        self.l0, self.l1 = layer(proj.layer0), layer(proj.layer1)
        n_freq = math.ceil(math.ceil(N_NEW / 2) / 2)
        n_in = n_freq * proj.layer1.conv.out_channels
        self.mid = nn.Linear(n_in, mid, bias=True) if mid else None
        self.out = nn.Linear(mid or n_in, hidden, bias=bias)
        nn.init.zeros_(self.out.weight)
        if bias:
            nn.init.zeros_(self.out.bias)

    def forward(self, high, mask=None):                       # high [B, T, 39]
        h = high.unsqueeze(1)
        for conv, norm in (self.l0, self.l1):
            if mask is not None:
                h = h * mask[:, None, :, None].to(h.dtype)
            h = F.relu(norm(conv(h).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            if mask is not None:
                mask = mask[:, ::2]
        B, _, T, _ = h.shape
        h = h.permute(0, 2, 3, 1).reshape(B, T, -1)
        if self.mid is not None:
            h = F.gelu(self.mid(h))
        return self.out(h)


def n_extra(n_old, pool):
    """how many high-end tokens follow n_old old ones: one per `pool` of his 40 ms tokens"""
    return -(-n_old // pool)


def pool_time(tok, n_old, pool):
    """[1, T, D] at 40 ms -> [1, n_extra, D]: the mean over each group of `pool` tokens (the last group may be short)"""
    tok = tok[:, :n_old]
    k = n_extra(n_old, pool)
    pad = k * pool - n_old
    s = F.pad(tok, (0, 0, 0, pad)).reshape(1, k, pool, -1).sum(2)
    cnt = torch.full((k,), float(pool), device=tok.device)
    cnt[-1] = pool - pad
    return s / cnt[None, :, None]


@torch.no_grad()
def silence_vector(proc, model):
    """what his ears hand on for silence (the mean over 2 s of zeros): the blank page his brain already knows"""
    z = np.zeros(32000, np.float32)
    f = proc.feature_extractor([z], sampling_rate=16000, return_tensors="pt")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.model.audio_tower(f["input_features"].to(model.device), f["input_features_mask"].to(model.device))
    return out.last_hidden_state[0][out.attention_mask[0]].float().mean(0)


def attach(model, where="early", mid=0, pool=4):
    """puts the side path on the model's ears; set model._high before each forward (None = old ears only).
    early: added to his input projection's output, so it passes through his 12 ear layers (which clamp their numbers
    to ranges set on speech); late: added to what his ears hand on (after those layers), before the step into his
    brain. Either way his ears' own weights are untouched.
    sep: the high end goes on its own page: extra audio tokens after his old ones (one per `pool` old tokens, inside
    the same <audio> ... </audio>), through his own step into the brain (embed_audio). His brain reads audio in order
    and never looks ahead, so his old tokens are exactly what they were; only what comes after can see the new ones.
    Day one the new tokens are all the same blank page (silence_vector, set by train), not zeros."""
    tower = model.model.audio_tower
    proj = tower.subsample_conv_projection
    dim = tower.config.hidden_size if where == "early" else tower.config.output_proj_dims
    side = Side(proj, dim, mid, bias=where == "sep").to(next(proj.parameters()).device)
    side.where, side.pool = where, pool
    if where == "sep":
        old_gaf = model.model.get_audio_features

        def gaf(input_features, input_features_mask=None, **kw):
            out = old_gaf(input_features, input_features_mask, **kw)
            hi = getattr(model, "_high", None)
            if hi is None:
                return out
            assert out.pooler_output.shape[0] == 1, "sep handles one clip at a time"
            keep = out.attention_mask[0].to(out.pooler_output.device)
            old = out.pooler_output[0][keep]                              # his old tokens, untouched
            tok = side(hi.to(old.device, torch.float32), input_features_mask)
            new = model.model.embed_audio(inputs_embeds=pool_time(tok, old.shape[0], pool).to(old.dtype))[0]
            out.pooler_output = torch.cat([old, new])[None]
            out.attention_mask = torch.ones(1, out.pooler_output.shape[1], dtype=torch.bool, device=keep.device)
            return out
        model.model.get_audio_features = gaf
    elif where == "early":
        old = proj.forward

        def forward(input_features, input_features_mask=None):
            h, m = old(input_features, input_features_mask)
            hi = getattr(model, "_high", None)
            if hi is not None:
                add = side(hi.to(h.device, torch.float32), input_features_mask)
                h = h + add[:, : h.shape[1]].to(h.dtype)
            return h, m
        proj.forward = forward
    else:
        old_t = tower.forward

        def tforward(input_features, attention_mask=None, **kw):
            out = old_t(input_features, attention_mask, **kw)
            hi = getattr(model, "_high", None)
            if hi is not None:
                h = out.last_hidden_state
                add = side(hi.to(h.device, torch.float32), attention_mask)
                out.last_hidden_state = h + add[:, : h.shape[1]].to(h.dtype)
            return out
        tower.forward = tforward
    model._side = side
    return side


LISTEN_TARGETS = re.compile(r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj)$")


def wear_listen(model, rank):
    """round 8 (Immy 2026-10-21, "something's bogging it down"; the probe: the tokens his brain receives carry the
    high end far better than he says it): his brain learns to listen too. A LoRA on his language rooms that acts ONLY
    on the audio positions of what he reads (his old audio tokens and the high-end page), never on text: Partial LoRA
    (BLSP-KD, Wang et al. 2024). Text-only prompts, and every word he writes, pass through untouched, so his talk can't
    change; only how he takes in sound does. model._listen_on switches it (the keep lesson's teacher takes it off).
    B starts at zero: day one he hears exactly as before."""
    if getattr(model, "_listen", None) is not None:
        return model._listen
    aid = model.config.audio_token_id

    def mark(mod, args, kwargs):                              # which positions of this forward are sound
        ids = kwargs.get("input_ids", args[0] if args else None)
        model._audio_mask = (ids == aid) if ids is not None else None
    model.register_forward_pre_hook(mark, with_kwargs=True)
    g = torch.Generator().manual_seed(8)
    pieces = {}
    for name, lin in model.named_modules():
        if not (isinstance(lin, nn.Linear) and LISTEN_TARGETS.match(name)):
            continue
        A = nn.Parameter((torch.randn(rank, lin.in_features, generator=g) / math.sqrt(lin.in_features)).to(lin.weight.device))
        B = nn.Parameter(torch.zeros(lin.out_features, rank, device=lin.weight.device))
        def hook(mod, inp, out, A=A, B=B):
            m = getattr(model, "_audio_mask", None)
            if not getattr(model, "_listen_on", False) or m is None or m.shape != out.shape[:2] or not m.any():
                return out
            return out + (((inp[0].float() @ A.T) @ B.T) * m[..., None]).to(out.dtype)
        lin.register_forward_hook(hook)
        pieces[name] = (A, B)
    model._listen, model._listen_on = pieces, True
    print(json.dumps({"listen": len(pieces), "rank": rank}), flush=True)
    return pieces


def load_side(model, side_pt):
    """attaches a saved side path the way it was trained (early or late, its hidden layer), or an empty early one;
    with its listener (round 8 on) when it has one"""
    if not side_pt:
        return attach(model)
    ck = torch.load(side_pt, map_location=model.device)
    side = attach(model, ck.get("where", "early"), ck.get("mid", 0), ck.get("pool", 4))
    side.load_state_dict(ck["side"])
    if ck.get("listen"):
        pieces = wear_listen(model, ck["listen_rank"])
        for name, (A, B) in pieces.items():
            A.data.copy_(ck["listen"][name + ".A"]); B.data.copy_(ck["listen"][name + ".B"])
    return side


def wear_talk(model, adapter_dir):
    """his talk adapter (round 3: LoRA on his language rooms only, never his ears), worn the way his live brain wears
    it: each target layer adds scale * x A^T B^T. model._talk_on switches it (the keep lesson's teacher takes it off).
    Why (2026-10-21): in his live brain the round 5-6 ears flipped IDC's hook to "it's not okay", and the same ears
    without this adapter kept it on every line; the adapter was trained on text only and never learned to copy words"""
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    assert not cfg.get("use_rslora") and not cfg.get("use_dora") and not cfg.get("rank_pattern")
    scale = cfg["lora_alpha"] / cfg["r"]
    w = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    mods = dict(model.named_modules())
    n = 0
    for k in w:
        if not k.endswith(".lora_A.weight"):
            continue
        lin = mods[k[len("base_model.model."):-len(".lora_A.weight")]]
        dev = lin.weight.device
        A, B = w[k].to(dev, torch.bfloat16), w[k.replace(".lora_A.", ".lora_B.")].to(dev, torch.bfloat16)

        def hook(mod, inp, out, A=A, B=B):
            if not model._talk_on:
                return out
            return out + ((inp[0].to(A.dtype) @ A.T) @ B.T * scale).to(out.dtype)
        lin.register_forward_hook(hook)
        n += 1
    model._talk_on = True
    print(json.dumps({"talk": adapter_dir, "layers": n, "scale": scale}), flush=True)
    return n


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


REACH_ANS = [str(k) for k in range(8, 18)] + ["past 18"]
AIR_ANS = [w for _, w in AIR]


def card_example(card, x, rng, balance=None):
    """one question about one clip; the clip may be lowpassed first (reach and air are measured after that).
    balance (the strict run, Immy 2026-10-20: "he can't play it safe"): every reach answer (8 ... 17 kHz, past 18)
    and every air answer comes up equally often, so the common middle answer stops paying: an answer is picked first,
    the cut made to produce it, and a clip that doesn't give it is skipped. balance is a running count per answer
    (one per worker): the answer asked for is always one of the least given so far, so rare ones (a lot of high end)
    catch up instead of coming out as rarely as they occur"""
    if np.abs(x).max() < 1e-3:
        return None                                           # a silent crop
    kind = rng.choice(["name", "reach", "reach", "air"] + (["air"] * 3 if balance is not None else []))
    # (balance: air asked more often, since waiting for its rare answers skips most clips)
    def least(answers):
        lo = min(balance.get(a, 0) for a in answers)
        return rng.choice([a for a in answers if balance.get(a, 0) == lo])
    if balance is not None and kind == "reach":
        want = least(REACH_ANS)
        cut = None if want == "past 18" else 8 if want == "8" else int(want)
        y = lowpass(x, cut)
        r = reach_khz(y)
        if r < 7:
            return None
        r = min(r, cut) if cut else r
        got = "past 18" if r > 18 else str(max(8, round(r)))
        if got != want:
            return None
        ans = "All the way up, past 18 kHz." if want == "past 18" else f"Up to about {want} kHz."
        balance[want] = balance.get(want, 0) + 1
        return y, Q[kind], ans, kind
    if balance is not None and kind == "air":
        want = least(AIR_ANS)
        y = lowpass(x, rng.choice([None, None, None, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
        if reach_khz(y) < 7 or next(w for t, w in AIR if air_share(y) < t) != want:
            return None
        balance[want] = balance.get(want, 0) + 1
        return y, Q[kind], want[0].upper() + want[1:] + ".", kind
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
        if ans == "almost none" and rng.random() < 0.6:
            return None                                       # half the clips are "almost none": keep 4 in 10
        ans = ans[0].upper() + ans[1:] + "."
    return y, Q[kind], ans, kind


def add_pages(enc, model, side_on):
    """with a sep side path on: room for the high-end tokens right after his old audio tokens (every per-token row,
    input_ids and its masks, gets copies of the last old audio token's entries). Returns how many were added."""
    side = getattr(model, "_side", None)
    if not side_on or side is None or side.where != "sep":
        return 0
    ids = enc["input_ids"][0]
    where = (ids == model.config.audio_token_id).nonzero()[:, 0]
    k = n_extra(len(where), side.pool)
    at = int(where[-1]) + 1
    shape = enc["input_ids"].shape
    for name, v in list(enc.items()):
        if not name.startswith("input_features") and v.dim() == 2 and v.shape == shape:
            enc[name] = torch.cat([v[:, :at], v[:, at - 1:at].expand(1, k), v[:, at:]], dim=1)
    return k


def encode(proc, model, high_fe, y48, question, answer=None, side_on=True):
    """the chat (sound, then the question), the old ears' features from the 16 kHz version, the new bands from the
    48 kHz one; with an answer: labels on the answer's tokens only"""
    y16 = resample_poly(y48, 1, 3).astype(np.float32)
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": y16}, {"type": "text", "text": question}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = prompt + (answer + "<turn|>" if answer is not None else "")
    enc = proc(text=[full], audio=[y16], return_tensors="pt")
    dev = model.device
    enc = {k: v.to(dev) for k, v in enc.items()}
    extra = add_pages(enc, model, side_on)
    T = enc["input_features"].shape[1]
    hi = high_fe(torch.from_numpy(y48).to(dev), T).unsqueeze(0)
    m = enc["input_features_mask"]
    hi = hi * m[..., None].to(hi.dtype)
    labels = None
    if answer is not None:
        tail = proc.tokenizer(answer + "<turn|>", add_special_tokens=False)["input_ids"]
        ids = enc["input_ids"][0]
        if ids[-len(tail):].tolist() != tail:                # the answer didn't tokenize on its own the same way
            n_prompt = proc(text=[prompt], audio=[y16], return_tensors="pt")["input_ids"].shape[1] + extra
        else:
            n_prompt = len(ids) - len(tail)
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = -100
    return enc, hi, labels


IDC = re.compile(r"(^|[^a-z])idc([^a-z]|$)", re.I)    # Immy's friend's track and its versions: the words test, never taught
ASK_WORDS = ["Transcribe the words in this audio.", "Transcribe the words in this audio. Write only the words.",
             "What are they saying?", "Write down the lyrics you hear.", "What words do you hear in this?"]


def cards_of(cards_dir, split):
    rows = [json.loads(l) for l in open(os.path.join(cards_dir, "cards.jsonl"))]
    return [r for r in rows if r["split"] == split and not (split == "train" and IDC.search(r["src"]))]


def keep_loss(proc, model, high_fe, y48, question, old_answer):
    """how far the side path moves his answer: KL from the old ears' next-word odds to the new ones, on the old answer's
    words (both read the same sound and the same answer; only the side path differs; with sep, the old ears' chat has
    no high-end page and the new one does, and the answer sits at the end of both)"""
    old, _, _ = encode(proc, model, high_fe, y48, question, old_answer, side_on=False)
    enc, hi, labels = encode(proc, model, high_fe, y48, question, old_answer)
    n = int((labels[0] != -100).sum()) + 1                    # the answer's words, and the step that predicts the first
    talk, listen = getattr(model, "_talk_on", None), getattr(model, "_listen_on", False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model._high = None
        model._talk_on = False                                # the teacher: his old ears, his brain without the adapter
        model._listen_on = False                              # (and without the listener)
        teach = model(**old, logits_to_keep=n).logits[0, :-1].float()
    model._talk_on, model._listen_on = talk, listen
    model._high = hi
    with torch.autocast("cuda", dtype=torch.bfloat16):
        student = model(**enc, logits_to_keep=n).logits[0, :-1].float()
    return F.kl_div(F.log_softmax(student, -1), F.log_softmax(teach, -1), log_target=True, reduction="batchmean")


def strict_loss(model, enc, labels, alpha):
    """the strict run (Immy: "you either say it's this or it's not"): the usual pull toward the right answer, plus
    unlikelihood (Welleck et al. 2019; Li et al. 2020): wherever his top guess for an answer word is wrong, that guess
    is pushed down, -log(1 - p(wrong guess)), weighted by alpha. A confident wrong answer (the safe middle one) costs
    the most. Returns (loss, the punishment part as a number)"""
    n = int((labels[0] != -100).sum()) + 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**enc, logits_to_keep=n).logits[0, :-1].float()
    gold = labels[0, -(n - 1):]
    logp = F.log_softmax(logits, -1)
    ce = F.nll_loss(logp, gold)
    guess = logp.argmax(-1)
    wrong = guess != gold
    p_guess = logp.gather(1, guess[:, None])[:, 0].exp()
    ul = (-torch.log1p(-p_guess.clamp(max=1 - 1e-4)) * wrong).sum() / max(1, len(gold))
    return ce + alpha * ul, float(ul)


def keepset(keep_dir, model_dir, *rest):
    kw = {rest[i][2:]: rest[i + 1] for i in range(0, len(rest), 2)}
    n = int(kw.get("n", 600))
    proc, model = load(model_dir)
    attach(model)
    high_fe = HighBands().to(model.device)
    rows = [r for r in (json.loads(l) for l in open(os.path.join(keep_dir, "cards.jsonl"))) if not IDC.search(r["src"])]
    rng = random.Random(11)
    rng.shuffle(rows)
    out = os.path.join(keep_dir, "keep.jsonl")
    done = {json.loads(l)["clip"] for l in open(out)} if os.path.exists(out) else set()
    with open(out, "a") as f:
        for r in rows[:n]:
            if r["clip"] in done:
                continue
            q = rng.choice(ASK_WORDS)
            a = answer(proc, model, high_fe, read_flac(os.path.join(keep_dir, "clips", r["clip"])), q, False, 80)
            f.write(json.dumps({"clip": r["clip"], "name": r["name"], "q": q, "a": a}) + "\n"); f.flush()


def train(cards_dir, model_dir, out_dir, *rest):
    kw = {rest[i][2:]: rest[i + 1] for i in range(0, len(rest), 2)}
    steps, lr, accum = int(kw.get("steps", 6000)), float(kw.get("lr", 3e-4)), int(kw.get("accum", 8))
    where, mid, pool = kw.get("where", "early"), int(kw.get("mid", 0)), int(kw.get("pool", 4))
    os.makedirs(out_dir, exist_ok=True)
    proc, model = load(model_dir)
    if kw.get("talk"):
        wear_talk(model, kw["talk"])
    side = attach(model, where, mid, pool)
    if kw.get("init"):                                        # build on an accepted round (Immy's fallback rule)
        side.load_state_dict(torch.load(kw["init"], map_location=model.device)["side"])
    elif where == "sep":
        side.out.bias.data.copy_(silence_vector(proc, model))
    strict, balance = float(kw.get("strict", 0)), kw.get("balance", "0") == "1"
    high_fe = HighBands().to(model.device)
    groups = [{"params": list(side.parameters()), "lr": lr}]
    listen_rank = int(kw.get("listen", 0))
    if listen_rank:                                           # round 8: his brain learns to listen (wear_listen)
        pieces = wear_listen(model, listen_rank)
        groups.append({"params": [p for ab in pieces.values() for p in ab], "lr": float(kw.get("listen_lr", 1e-4))})
    trainable = [p for g_ in groups for p in g_["params"]]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    cards = cards_of(cards_dir, "train")
    log = open(os.path.join(out_dir, "train.jsonl"), "a")
    seen, run = 0, {}
    # the clips are read, cut and measured by worker processes while the GPU trains
    from torch.utils.data import DataLoader, IterableDataset

    class Lessons(IterableDataset):
        def __iter__(self):
            w = torch.utils.data.get_worker_info()
            rng = random.Random(1000 + (w.id if w else 0))
            counts = {} if balance else None              # this worker's answers so far (balance)
            while True:
                c = rng.choice(cards)
                ex = card_example(c, read_flac(os.path.join(cards_dir, "clips", c["clip"])), rng, counts)
                if ex is not None:
                    yield ex
    lessons = iter(DataLoader(Lessons(), batch_size=None, num_workers=4, prefetch_factor=8))
    keep_dir, keep_every = kw.get("keep"), int(kw.get("keep_every", 3))
    keeps = [json.loads(l) for l in open(os.path.join(keep_dir, "keep.jsonl"))] if keep_dir else []
    krng = random.Random(5)
    import time
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        done = 0
        while done < accum:
            if keeps and (seen % keep_every) == keep_every - 1:
                k = krng.choice(keeps)
                loss = keep_loss(proc, model, high_fe, read_flac(os.path.join(keep_dir, "clips", k["clip"])), k["q"], k["a"])
                kind = "keep"
            else:
                y, q, a, kind = next(lessons)
                y = np.asarray(y, dtype=np.float32)
                enc, hi, labels = encode(proc, model, high_fe, y, q, a)
                model._high = hi
                if strict > 0:
                    loss, ul = strict_loss(model, enc, labels, strict)
                    run.setdefault("punished", []).append(ul)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(**enc, labels=labels).loss
            (loss / accum).backward()
            run.setdefault(kind, []).append(loss.item())
            done += 1; seen += 1
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step()
        if (step + 1) % 50 == 0:
            row = {"step": step + 1, "seen": seen, "lr": sched.get_last_lr()[0], "sec_per_step": round((time.time() - t0) / (step + 1), 2),
                   **{k: round(float(np.mean(v)), 4) for k, v in run.items()}}
            log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
            run = {}
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            if listen_rank:
                listen_sd = {f"{n}.{w}": t.detach().cpu() for n, (A, B) in pieces.items() for w, t in (("A", A), ("B", B))}
            ck = {"listen": listen_sd if listen_rank else None, "listen_rank": listen_rank,
                  "side": side.state_dict(), "edges_hz": high_fe.edges.tolist(), "step": step + 1, "where": where,
                  "mid": mid, "pool": pool, "init": kw.get("init"), "strict": strict, "balance": balance,
                  "talk": kw.get("talk")}
            torch.save(ck, os.path.join(out_dir, f"side-{step + 1}.pt"))
            torch.save(ck, os.path.join(out_dir, "side.pt"))


@torch.no_grad()
def answer(proc, model, high_fe, y48, question, use_side, max_new=40):
    enc, hi, _ = encode(proc, model, high_fe, y48, question, side_on=use_side)
    model._high = hi if use_side else None
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def eval_(cards_dir, model_dir, out, *rest):
    kw = {rest[i][2:]: rest[i + 1] for i in range(0, len(rest), 2)}
    n = int(kw.get("n", 300))
    proc, model = load(model_dir)
    if kw.get("talk"):
        wear_talk(model, kw["talk"])
    side = load_side(model, kw.get("side"))
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
    side_pt, talk, asks, i = None, None, [], 0
    while i < len(rest):
        if rest[i] == "--side": side_pt = rest[i + 1]
        elif rest[i] == "--talk": talk = rest[i + 1]
        elif rest[i] == "--ask": asks.append(rest[i + 1])
        i += 2
    asks = asks or ["What do you hear in this? Describe the sound, including the high end."]
    proc, model = load(model_dir)
    if talk:
        wear_talk(model, talk)
    side = load_side(model, side_pt)
    high_fe = HighBands().to(model.device)
    import soundfile as sf
    x, sr = sf.read(wav, dtype="float32", always_2d=False)
    x = x.mean(1) if x.ndim > 1 else x
    x = resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x
    x = x[: 30 * SR]
    res = [{"ask": q, "side": side_pt, "talk": talk, "answer": answer(proc, model, high_fe, x, q, side_pt is not None, 120)} for q in asks]
    json.dump(res, open(out, "x"), indent=1)
    print(json.dumps(res, indent=1))


@torch.no_grad()
def pagecheck(model_dir, wav, out, *rest):
    """sep before any training: (1) his brain's states over everything up to his last old audio token, without and
    with the high-end page (should be the same: nothing there can see the page); (2) the transcription both ways;
    (3) on --keep clips, how far an untrained page moves his answers (KL, as in the keep lesson) when the page is the
    silence vector vs zeros"""
    kw = {rest[i][2:]: rest[i + 1] for i in range(0, len(rest), 2)}
    proc, model = load(model_dir)
    side = attach(model, "sep", int(kw.get("mid", 1024)), int(kw.get("pool", 4)))
    high_fe = HighBands().to(model.device)
    silence = silence_vector(proc, model)
    side.out.bias.data.copy_(silence)
    import soundfile as sf
    x, sr = sf.read(wav, dtype="float32", always_2d=False)
    x = x.mean(1) if x.ndim > 1 else x
    x = (resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x)[: 30 * SR]
    q = "Transcribe the words in this audio. Write only the words."
    old, _, _ = encode(proc, model, high_fe, x, q, side_on=False)
    new, hi, _ = encode(proc, model, high_fe, x, q)
    last = int((old["input_ids"][0] == model.config.audio_token_id).nonzero()[-1]) + 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model._high = None
        h_old = model(**old, output_hidden_states=True).hidden_states
        model._high = hi
        h_new = model(**new, output_hidden_states=True).hidden_states
    k = new["input_ids"].shape[1] - old["input_ids"].shape[1]
    # the control: the same k tokens added at the very end, where by construction nothing before can see them; any
    # difference there is the GPU rounding differently for a longer sequence, not the page leaking
    ctl = {n: (torch.cat([v, v[:, -1:].expand(1, k)], 1) if not n.startswith("input_features") and v.dim() == 2
               and v.shape == old["input_ids"].shape else v) for n, v in old.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model._high = None
        h_ctl = model(**ctl, output_hidden_states=True).hidden_states

    def rel(hs):                                              # worst layer: |difference| / |state|, over the old part
        return max(float((a[0, :last] - b[0, :last]).float().norm() / a[0, :last].float().norm())
                   for a, b in zip(h_old, hs))
    res = {"wav": wav, "old_tokens": last, "page_tokens": k,
           "before_page_rel_diff_page": rel(h_new), "before_page_rel_diff_control": rel(h_ctl),
           "words_old": answer(proc, model, high_fe, x, q, False, 200),
           "words_day_one": answer(proc, model, high_fe, x, q, True, 200)}
    if kw.get("keep"):
        ks = [json.loads(l) for l in open(os.path.join(kw["keep"], "keep.jsonl"))][: int(kw.get("n", 20))]
        for name, b in (("silence", silence), ("zeros", torch.zeros_like(silence))):
            side.out.bias.data.copy_(b)
            res["keep_kl_" + name] = round(float(np.mean([keep_loss(proc, model, high_fe, read_flac(
                os.path.join(kw["keep"], "clips", k["clip"])), k["q"], k["a"]).item() for k in ks])), 4)
    json.dump(res, open(out, "x"), indent=1)
    print(json.dumps(res, indent=1))


def wear_fix(model, adapter_dir, rank):
    """a new LoRA piece on the same layers as his talk adapter, worn on top of it (model._fix_on switches it). B starts
    at zero, so on day one he is exactly round 3. Trained in fp32 and saved in the talk adapter's own format, so
    tools/train/distill.py join makes one adapter of both for his brainstem"""
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    from safetensors.torch import load_file
    names = [k[len("base_model.model."):-len(".lora_A.weight")]
             for k in load_file(os.path.join(adapter_dir, "adapter_model.safetensors")) if k.endswith(".lora_A.weight")]
    mods = dict(model.named_modules())
    g = torch.Generator().manual_seed(7)
    fix = {}
    for name in names:
        lin = mods[name]
        A = nn.Parameter((torch.randn(rank, lin.in_features, generator=g) / math.sqrt(lin.in_features)).to(lin.weight.device))
        B = nn.Parameter(torch.zeros(lin.out_features, rank, device=lin.weight.device))
        def hook(mod, inp, out, A=A, B=B):
            if not model._fix_on:
                return out
            return out + ((inp[0].float() @ A.T) @ B.T).to(out.dtype)
        lin.register_forward_hook(hook)
        fix[name] = (A, B)
    model._fix_on = False
    return fix, dict(cfg, r=rank, lora_alpha=rank)


def save_fix(fix, cfg, out_dir):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    w = {}
    for name, (A, B) in fix.items():
        w[f"base_model.model.{name}.lora_A.weight"] = A.detach().to(torch.bfloat16).cpu().contiguous()
        w[f"base_model.model.{name}.lora_B.weight"] = B.detach().to(torch.bfloat16).cpu().contiguous()
    save_file(w, os.path.join(out_dir, "adapter_model.safetensors"))
    json.dump(cfg, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=1)


def word_match(said, ref):
    """share of the reference's words he says, in order (difflib's matching blocks over lowercased words)"""
    import difflib
    a, b = re.findall(r"[a-z']+", said.lower()), re.findall(r"[a-z']+", ref.lower())
    if not b:
        return 1.0 if not a else 0.0
    return sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()) / len(b)


def hearfix(keep_dir, model_dir, out_dir, *rest):
    """the talk side of the hook flip (2026-10-21, Immy: "if you want to fix him with the talking side as well, go
    ahead"). His talk adapter (round 3) was trained on text only, so with strong ears it rewrites what he hears (IDC:
    "bitch I don't care" -> "it's not okay"); without it his brain copies the words. A small new piece on top of round 3
    learns two things at once, both output matching, no labels and no AI judge:
      hear  on vocal clips (KEEP_DIR, never IDC): his transcription with the adapter on should be what his brain says
            with it off. The teacher's own greedy answer is the text (Wang et al. 2025, "Teacher CE"), and the loss is
            the KL from the teacher's next-word odds to his on those words (as the keep lesson)
      talk  on round 3's own rows (--text, distill.py's rows-r3.jsonl): the KL to round 3 as it is, so how he talks
            stays round 3's (the anchor distill.py uses)
    Others found the same: a speech LLM loses skills when fine-tuned on one modality and keeps them when the other
    modality's own outputs are mixed back in (DiVA, Held et al. 2025; VoiceTextBlender, Peng et al. 2024).
    Known gap: his neurons-off and mood drip hooks (tools/train/choices.py Brain) are not worn here, as in the ears
    rounds; the engine with his real brain is the verdict.
      hearfix KEEP_DIR MODEL OUT_DIR --talk ADAPTER --text rows-r3.jsonl [--side SIDE.pt] [--rank 8] [--steps 400]
              [--accum 8] [--lr 1e-4] [--text_w 1.0] [--hold 60] [--max_prompt 3072] [--max_words 400]
    writes OUT_DIR/teacher.jsonl (the teacher's answers, resumable), train.jsonl, verdict.json, fix/ (the new piece)"""
    import time
    kw = {rest[i][2:]: rest[i + 1] for i in range(0, len(rest), 2)}
    rank, steps, accum = int(kw.get("rank", 8)), int(kw.get("steps", 400)), int(kw.get("accum", 8))
    lr, text_w, hold = float(kw.get("lr", 1e-4)), float(kw.get("text_w", 1.0)), int(kw.get("hold", 60))
    max_prompt, max_words = int(kw.get("max_prompt", 3072)), int(kw.get("max_words", 400))
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "train.jsonl"), "a")
    say = lambda **r: (print(json.dumps(r), flush=True), log.write(json.dumps(r) + "\n"), log.flush())
    proc, model = load(model_dir)
    wear_talk(model, kw["talk"])
    fix, cfg = wear_fix(model, kw["talk"], rank)
    side = load_side(model, kw.get("side"))
    use_side = kw.get("side") is not None
    high_fe = HighBands().to(model.device)
    tok = proc.tokenizer

    # the teacher's answers: his brain without the talk adapter, these ears, greedy, long enough for a verse
    keeps = [json.loads(l) for l in open(os.path.join(keep_dir, "keep.jsonl"))]
    keeps = [k for k in keeps if not IDC.search(k.get("src", "") + k["clip"])]
    tpath = os.path.join(out_dir, "teacher.jsonl")
    have = {json.loads(l)["clip"]: json.loads(l) for l in open(tpath)} if os.path.exists(tpath) else {}
    model._talk_on, model._fix_on = False, False
    with open(tpath, "a") as f:
        for k in keeps:
            if k["clip"] not in have:
                y = read_flac(os.path.join(keep_dir, "clips", k["clip"]))
                have[k["clip"]] = dict(k, teacher=answer(proc, model, high_fe, y, k["q"], use_side, 160))
                f.write(json.dumps(have[k["clip"]]) + "\n"); f.flush()
    teach = [have[k["clip"]] for k in keeps if have[k["clip"]]["teacher"]]
    rng = random.Random(21)
    rng.shuffle(teach)
    hear_ho, hear_tr = teach[:hold], teach[hold:]
    rows = [json.loads(l) for l in open(kw["text"])]
    text_tr = [r for r in rows if not r["held_out"]]
    text_ho = [r for r in rows if r["held_out"]][:100]

    def hear_kl(t, grad):
        y = read_flac(os.path.join(keep_dir, "clips", t["clip"]))
        enc, hi, labels = encode(proc, model, high_fe, y, t["q"], t["teacher"], side_on=use_side)
        n = int((labels[0] != -100).sum()) + 1
        model._high = hi if use_side else None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model._talk_on, model._fix_on = False, False
            ref = model(**enc, logits_to_keep=n).logits[0, :-1].float()
        model._talk_on, model._fix_on = True, True
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            cur = model(**enc, logits_to_keep=n).logits[0, :-1].float()
        return F.kl_div(F.log_softmax(cur, -1), F.log_softmax(ref, -1), log_target=True, reduction="batchmean")

    def talk_kl(r, grad):
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"][-max_prompt:]
        t = tok(r["text"] + r["end"], add_special_tokens=False)["input_ids"][:max_words]
        x = torch.tensor([p + t], device=model.device)
        n = len(t) + 1
        model._high = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model._talk_on, model._fix_on = True, False
            ref = model(input_ids=x, logits_to_keep=n).logits[0, :-1].float()
        model._talk_on, model._fix_on = True, True
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            cur = model(input_ids=x, logits_to_keep=n).logits[0, :-1].float()
        return F.kl_div(F.log_softmax(cur, -1), F.log_softmax(ref, -1), log_target=True, reduction="batchmean")

    @torch.no_grad()
    def check(tag):
        """held-out clips: how much of the teacher's words he says with the adapter (and the fix) on; held-out talk:
        how far the fix moved round 3"""
        res = {"at": tag, "hear_kl": 0.0, "talk_kl": 0.0}
        said = []
        for t in hear_ho:
            res["hear_kl"] += float(hear_kl(t, False)) / len(hear_ho)
            model._talk_on, model._fix_on = True, True
            y = read_flac(os.path.join(keep_dir, "clips", t["clip"]))
            a = answer(proc, model, high_fe, y, t["q"], use_side, 160)
            said.append({"clip": t["clip"], "q": t["q"], "teacher": t["teacher"], "said": a,
                         "match": round(word_match(a, t["teacher"]), 3)})
        for r in text_ho:
            res["talk_kl"] += float(talk_kl(r, False)) / len(text_ho)
        res["hear_match"] = round(float(np.mean([s["match"] for s in said])), 3)
        res["hear_exact"] = round(float(np.mean([s["match"] == 1.0 for s in said])), 3)
        res = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}
        json.dump(said, open(os.path.join(out_dir, f"said-{tag}.json"), "w"), indent=1)
        return res

    before = check("before")                                 # fix at zero: round 3 as it is live
    say(mode="hearfix", talk=kw["talk"], side=kw.get("side"), rank=rank, hear_train=len(hear_tr),
        hear_held=len(hear_ho), text_train=len(text_tr), text_held=len(text_ho), before=before)
    params = [p for A_B in fix.values() for p in A_B]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    t0, run, seen = time.time(), {"hear": [], "talk": []}, 0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for j in range(accum):                                # one sound row, one talk row, in turn
            if seen % 2 == 0:
                loss = hear_kl(rng.choice(hear_tr), True); run["hear"].append(loss.item())
            else:
                loss = text_w * talk_kl(rng.choice(text_tr), True); run["talk"].append(loss.item() / text_w)
            (loss / accum).backward()
            seen += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if (step + 1) % 20 == 0:
            say(step=step + 1, lr=sched.get_last_lr()[0], sec_per_step=round((time.time() - t0) / (step + 1), 2),
                **{k: round(float(np.mean(v)), 4) for k, v in run.items() if v})
            run = {"hear": [], "talk": []}
    save_fix(fix, cfg, os.path.join(out_dir, "fix"))
    after = check("after")
    json.dump({"talk": kw["talk"], "side": kw.get("side"), "before": before, "after": after},
              open(os.path.join(out_dir, "verdict.json"), "w"), indent=1)
    say(done=True, before=before, after=after)


def listenout(side_pt, out_dir):
    """the listener (round 8, --listen) out of side.pt as an ordinary peft adapter folder, for convert_lora_to_gguf.py.
    It adds x A^T B^T with no scale, so lora_alpha = r. The engine wears it only while reading a sound (lg-mind's
    LG_LISTEN); worn on text it would be a different brain from the one trained"""
    ck = torch.load(side_pt, map_location="cpu")
    if "listen" not in ck:
        raise SystemExit(f"{side_pt} has no listener (trained without --listen)")
    L, r = ck["listen"], int(ck["listen_rank"])
    pieces = {k[:-2]: (L[k], L[k[:-2] + ".B"]) for k in L if k.endswith(".A")}
    save_fix(pieces, {"peft_type": "LORA", "r": r, "lora_alpha": r, "lora_dropout": 0.0, "bias": "none",
                      "use_rslora": False, "use_dora": False, "fan_in_fan_out": False,
                      "target_modules": LISTEN_TARGETS.pattern.rstrip("$")}, out_dir)
    print(f"wrote {out_dir}: {len(pieces)} pieces, rank {r}")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"train": train, "eval": eval_, "listen": listen, "keepset": keepset, "pagecheck": pagecheck,
     "hearfix": hearfix, "listenout": listenout}[cmd](*args)
