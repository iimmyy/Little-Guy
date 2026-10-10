"""Pair meter (Immy's 11-and-17 idea): which FFN detectors take turns, so seeing one audible tells you its partner is muted?

Learn: on his first 40 chats, record which detectors are audible (-40 dB) on every word he wrote, and count how often
each pair is audible together. A "clean partner" B of A: both are regulars (audible on 5%+ of words), yet B is
audible on under 1% of the words where A is.
Use: on 10 unseen chats, meter a set of anchor detectors per word (the ones with the most clean partners), then
mute every detector that is a clean partner of an audible anchor. Report how many lanes the table mutes per word
without playing them, and how many of those were actually audible (mistakes) and how loud the mistakes were.

Brackets (Immy's grouping idea): regrouping detectors changes nothing he says, so sort each layer's detectors into
64 folders of ones that play together (k-means on when each is audible, learned on the first 40 chats). Then, per
unseen word: how many lanes must be streamed if whole folders are played until 95% of that word's FFN sound is
covered? Compared with folders of random detectors (no grouping) and with picking single lanes perfectly.

Usage: python pair_meter.py [ANCHOR_PERCENT ...]      e.g. python pair_meter.py 5 10 20
"""
import re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / ".run/weights/masters/e4b-qat"
CHATS = ROOT / ".run/calib/chat_calib.txt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ON_DB, REGULAR, CLEAN = 40, 0.05, 0.01
FOLDERS, COVER = 64, 0.95
LEARN, TEST = range(0, 40), range(40, 50)

state = {"start": 0, "on": {}, "level": {}}


def meter_forward(self, x):
    out = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
    lv = out[0, state["start"]:].float().abs()
    lv = lv / lv.amax(dim=-1, keepdim=True).clamp_min(1e-12)       # loudness relative to the loudest, per word
    state["on"].setdefault(self.idx, []).append((lv >= 10 ** (-ON_DB / 20)).to(torch.uint8).cpu())
    state["level"].setdefault(self.idx, []).append(lv.half().cpu())
    return self.down_proj(out)


def record(model, tok, chats):
    state.update(on={}, level={})
    for chat in chats:
        for r in re.finditer(r"<\|turn>model\n.*?<turn\|>", chat, re.S):
            text = chat[:r.end()]
            enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
            reply_char = r.start() + len("<|turn>model\n")
            start = next(i for i, (a, b) in enumerate(enc.offset_mapping[0].tolist()) if a >= reply_char and b > a)
            ids = enc.input_ids[:, -1024:]
            state["start"] = start - (enc.input_ids.shape[1] - ids.shape[1])
            model(ids.to(DEVICE), logits_to_keep=1)
    on = {i: torch.cat(v) for i, v in state["on"].items()}
    level = {i: torch.cat(v) for i, v in state["level"].items()}
    return on, level


def kmeans(x, k, iters=25):
    """Group the rows of x (detectors x words) into k folders of detectors that are audible on the same words."""
    g = torch.Generator().manual_seed(0)
    c = x[torch.randperm(x.shape[0], generator=g)[:k].to(x.device)].clone()
    for _ in range(iters):
        label = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = label == j
            if m.any():
                c[j] = x[m].mean(0)
    return label


def lanes_streamed(energy, label, k):
    """Per word: play whole folders, loudest first, until COVER of the word's sound is in; share of lanes streamed."""
    onehot = F.one_hot(label, k).float()                            # detectors x folders
    fe = energy @ onehot                                             # words x folders
    order = fe.argsort(1, descending=True)
    cum = fe.gather(1, order).cumsum(1) / fe.sum(1, keepdim=True)
    n = (cum < COVER).sum(1) + 1                                     # folders needed per word
    size = onehot.sum(0)[order]
    streamed = (size.cumsum(1).gather(1, (n - 1)[:, None]))[:, 0]
    return float((streamed / label.numel()).mean())


def main():
    anchor_pcts = [int(a) for a in sys.argv[1:]] or [5, 10, 20]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()
    mlps = [m for m in model.modules() if type(m).__name__ == "Gemma4TextMLP"]
    for i, m in enumerate(mlps):
        m.idx = i
    type(mlps[0]).forward = meter_forward
    chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()]
    with torch.no_grad():
        learn_on, _ = record(model, tok, [chats[i] for i in LEARN])
        print(f"learned on {len(LEARN)} chats, {learn_on[0].shape[0]} of his words [{time.time() - t0:.0f} s]", flush=True)
        test_on, test_level = record(model, tok, [chats[i] for i in TEST])
        print(f"tested on {len(TEST)} unseen chats, {test_on[0].shape[0]} words [{time.time() - t0:.0f} s]", flush=True)
    del model
    torch.mps.empty_cache()

    has_partner, results, brackets = [], {a: [] for a in anchor_pcts}, []
    for i in sorted(learn_on):
        A = learn_on[i].to(DEVICE, torch.float16)                      # words x detectors
        count = A.sum(0).float()
        together = (A.T @ A).float()                                   # how often each pair is audible together
        regular = count >= REGULAR * A.shape[0]
        clean = (together / count.clamp_min(1)[:, None] < CLEAN) & regular[:, None] & regular[None, :]
        n_partners = clean.sum(1)
        has_partner.append(float((n_partners > 0).float()[regular].mean()))
        for a in anchor_pcts:
            anchors = n_partners.topk(int(a / 100 * A.shape[1])).indices
            on = test_on[i].to(DEVICE, torch.float16)
            muted = (on[:, anchors] @ clean[anchors].half()) > 0       # partner of any audible anchor
            muted[:, anchors] = False
            truly_on = on.bool()
            lv = test_level[i].to(DEVICE).float()
            mistakes = muted & truly_on
            results[a].append((float(muted.float().mean()),
                               float(mistakes.sum() / muted.sum().clamp_min(1)),
                               float(lv[mistakes].mean()) if mistakes.any() else 0.0))
        energy = test_level[i].to(DEVICE).float().pow(2)
        label = kmeans(A.T.float(), FOLDERS)
        rand = torch.randperm(A.shape[1], generator=torch.Generator().manual_seed(1)).to(DEVICE) % FOLDERS
        srt = energy.sort(1, descending=True).values
        single = float((((srt.cumsum(1) / srt.sum(1, keepdim=True)) < COVER).sum(1) + 1).float().mean() / A.shape[1])
        brackets.append((lanes_streamed(energy, label, FOLDERS), lanes_streamed(energy, rand, FOLDERS), single))
        del A, together, clean
        torch.mps.empty_cache()

    print(f"\nregular detectors with at least one clean opposite partner: {100 * sum(has_partner) / len(has_partner):.1f}% "
          f"(per layer from {100 * min(has_partner):.0f}% to {100 * max(has_partner):.0f}%)")
    print(f"\n{'anchors metered':>15} | {'lanes muted by the table':>24} | {'of those, actually audible':>26} | {'their loudness':>14}")
    for a, rows in results.items():
        m = sum(r[0] for r in rows) / len(rows); e = sum(r[1] for r in rows) / len(rows); l = sum(r[2] for r in rows) / len(rows)
        print(f"{a:14d}% | {100 * m:23.1f}% | {100 * e:25.1f}% | {20 * torch.log10(torch.tensor(max(l, 1e-6))).item():11.1f} dB")
    b = [sum(r[j] for r in brackets) / len(brackets) for j in range(3)]
    print(f"\nbrackets ({FOLDERS} folders per layer), lanes streamed per word to cover {COVER:.0%} of its FFN sound:")
    print(f"  grouped by playing together: {100 * b[0]:.1f}%   random folders: {100 * b[1]:.1f}%   perfect single lanes: {100 * b[2]:.1f}%")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
