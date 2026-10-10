"""Section muting (Immy's session-template idea; GRIFFIN, Dong et al. 2024): meter Immy's message, keep only the FFN
detectors it lit up, and mute the rest for his whole reply. Also reads two more meters in the same run:
  stems:          per word, how many detectors are audible (-40 dB) vs how many the whole reply uses
                  (a small per-word share inside a big reply union = detectors take turns)
  count from 11:  how much the audible set overlaps from one word to the next (warm start)

For every reply in his unseen chats (41 onward), pass 1 plays the full brain and records everything; then each mute
setting replays the chat with the prompt at full brain and his reply with only the kept detectors. Scored on his
reply's words only.
  oracle:   keep the detectors his reply actually used most (needs the future; the best any picker could do)
  message:  keep what Immy's latest message lit up
  prompt:   keep what the whole prompt lit up (persona + history + message)

Usage: python section_mute.py [N_CHATS]
"""
import math, re, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / ".run/weights/masters/e4b-qat"
CHATS = ROOT / ".run/calib/chat_calib.txt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ON_DB = 40
SETTINGS = [("oracle", 0.5), ("message", 0.5), ("prompt", 0.5), ("oracle", 0.3), ("message", 0.3), ("prompt", 0.3)]

state = {"mode": None, "start": 0, "msg": 0, "keep": {}, "rec": {}}


def section_forward(self, x):
    """Gemma4TextMLP.forward that can record each detector's output, or mute the unkept ones from `start` on."""
    out = F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
    if state["mode"] == "record":
        a = out[0].float()
        s, m = state["start"], state["msg"]
        z = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)            # each word's detector pattern, unit loudness
        reply = a[s:]
        on = reply.abs() >= reply.abs().amax(dim=-1, keepdim=True) * 10 ** (-ON_DB / 20)
        inter = (on[1:] & on[:-1]).sum(-1).float()
        union = (on[1:] | on[:-1]).sum(-1).float().clamp_min(1)
        state["rec"][self.idx] = {
            "prompt": z[:s].pow(2).sum(0),                                # GRIFFIN's score: summed squared pattern
            "message": z[m:s].pow(2).sum(0),
            "energy": reply.pow(2).sum(0),                                # what his reply actually played
            "per_word": on.float().mean(),
            "union": on.any(0).float().mean(),
            "next_word": (inter / union).mean() if len(on) > 1 else torch.tensor(float("nan")),
        }
    elif state["mode"] == "mute":
        out = out.clone()
        out[:, state["start"]:] *= state["keep"][self.idx]
    return self.down_proj(out)


def reply_loss(model, ids, start):
    n = ids.shape[1] - start                                              # his words, scored from the word before them
    logits = model(ids, logits_to_keep=n + 1).logits[0, :-1].float()
    return F.cross_entropy(logits, ids[0, start:], reduction="sum").item(), n


def main():
    n_chats = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MASTER)
    model = AutoModelForCausalLM.from_pretrained(MASTER, dtype=torch.bfloat16).to(DEVICE).eval()
    mlps = [m for m in model.modules() if type(m).__name__ == "Gemma4TextMLP"]
    for i, m in enumerate(mlps):
        m.idx = i
    type(mlps[0]).forward = section_forward
    print(f"loaded on {DEVICE} in {time.time() - t0:.0f} s, {len(mlps)} FFN layers", flush=True)

    chats = [p for p in re.split(r"(?=<\|turn>system)", CHATS.read_text()) if p.strip()][40:40 + n_chats]
    loss = {k: 0.0 for k in ["full"] + [f"{p} {f:.0%}" for p, f in SETTINGS]}
    kept_energy = {f"{p} {f:.0%}": [] for p, f in SETTINGS}
    meters = {"per_word": [], "union": [], "next_word": []}
    words = replies = 0
    with torch.no_grad():
        for c, chat in enumerate(chats):
            for r in re.finditer(r"<\|turn>model\n.*?<turn\|>", chat, re.S):
                text = chat[:r.end()]
                msg_char = text.rfind("<|turn>user\n") + len("<|turn>user\n")
                reply_char = r.start() + len("<|turn>model\n")
                enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
                offs = enc.offset_mapping[0].tolist()
                start = next(i for i, (a, b) in enumerate(offs) if a >= reply_char and b > a)
                msg = next(i for i, (a, b) in enumerate(offs) if a >= msg_char and b > a)
                ids = enc.input_ids[:, -1024:].to(DEVICE)
                cut = enc.input_ids.shape[1] - ids.shape[1]
                start, msg = start - cut, max(msg - cut, 0)

                state.update(mode="record", start=start, msg=msg, rec={})
                l, n = reply_loss(model, ids, start)
                loss["full"] += l; words += n; replies += 1
                rec = {i: {k: v.cpu() for k, v in d.items()} for i, d in state["rec"].items()}
                for k in meters:
                    meters[k] += [float(d[k]) for d in rec.values()]

                for picker, frac in SETTINGS:
                    keep, share = {}, []
                    for i, d in rec.items():
                        score = d["energy"] if picker == "oracle" else d[picker]
                        top = score.topk(int(frac * score.numel())).indices
                        mask = torch.zeros_like(score); mask[top] = 1
                        share.append(float((d["energy"] * mask).sum() / d["energy"].sum().clamp_min(1e-12)))
                        keep[i] = mask.to(DEVICE, torch.bfloat16)
                    state.update(mode="mute", keep=keep)
                    l, _ = reply_loss(model, ids, start)
                    loss[f"{picker} {frac:.0%}"] += l
                    kept_energy[f"{picker} {frac:.0%}"].append(sum(share) / len(share))
                state["mode"] = None
                torch.mps.empty_cache()
            base = math.exp(loss["full"] / words)
            row = "  ".join(f"{k} {100 * (math.exp(v / words) / base - 1):+.1f}%" for k, v in loss.items() if k != "full")
            print(f"chat {c + 1}/{len(chats)} ({replies} replies, {words} words) [{time.time() - t0:.0f} s]  {row}", flush=True)

    base = math.exp(loss["full"] / words)
    print(f"\n{replies} replies, {words} of his words. Full brain perplexity on his replies: {base:.3f}")
    print(f"{'setting':>14} | {'reply perplexity':>16} | {'his reply sound kept':>20}")
    for k, v in loss.items():
        if k == "full":
            continue
        p = math.exp(v / words)
        print(f"{k:>14} | {p:9.3f} ({100 * (p / base - 1):+5.1f}%) | {100 * sum(kept_energy[k]) / len(kept_energy[k]):19.1f}%")
    nan_free = [x for x in meters["next_word"] if x == x]
    print(f"\nstems: per word {100 * sum(meters['per_word']) / len(meters['per_word']):.1f}% of detectors audible (-{ON_DB} dB), "
          f"whole reply uses {100 * sum(meters['union']) / len(meters['union']):.1f}%")
    print(f"count from 11: word-to-word overlap of the audible set {100 * sum(nan_free) / len(nan_free):.1f}%")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
