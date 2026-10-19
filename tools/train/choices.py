#!/usr/bin/env python3
"""Training his one-word decisions (decisions.py rows) on a GPU: his brain as it runs on the Mac (the 160 neurons
switched off, his feelings dripped into rooms 18-40 the same way the brainstem does), small LoRA adapters, and for each
decision the exact expected score over his choice odds, sum over options of p(option) x what it earns (all-options
policy gradient: every option's reward is known, so nothing waits on a lucky draw). Two guard rails keep him himself:
his choice odds may not drift far from his untrained odds (KL), and on his own words (thoughts, replies) every word's
odds stay close to his untrained self (KL, the anchor).

  check  OUT                        his untrained odds here vs the brainstem's (grid 4-bit on the Mac): the same brain?
  train  OUT [--init ACCEPTED_ADAPTER] [--epochs 2] [--lr 1e-5] [--rank 16] [--kl 2.0] [--anchor 0.2] [--scale 10]
         ends with OUT/verdict.json against the model it started from: better / mild regression / deep regression
         (Immy's rule: a deep one is thrown away and the next round starts from the last accepted model)
  eval   OUT [--adapter DIR]        expected score on held-out days, gate and act separately, and how far he moved
common: --model HF_DIR --data decisions.jsonl --dials dials.npz --cal calibration.json --off neurons_off.json
        --holdout day-018,day-019 (days kept out of training, for eval)
Writes OUT/log.jsonl, OUT/adapter/ (train), OUT/eval*.json. Needs torch, transformers (Gemma 4), peft.
"""
import argparse, json, math, os, random, time
import numpy as np
import torch

DRIP = range(18, 41)                      # rooms his feelings drip into (tools/mind/brain.py)


def load(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    return tok, model


def text_layers(model):
    """the 42 text rooms (Gemma 4 keeps them under language_model; vision and audio towers have their own)"""
    best = None
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and "language_model" in name:
            best = mod
    if best is None:
        for name, mod in model.named_modules():
            if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) == 42:
                best = mod
    return best


class Brain:
    """hooks that make the HF model his brain: neurons off (down_proj input zeroed, as when baked) and the drip"""
    def __init__(self, model, a):
        self.layers = text_layers(model)
        d, cal = np.load(a.dials), json.load(open(a.cal))
        dev = next(model.parameters()).device
        self.dial = {k: torch.tensor(d[k], dtype=torch.float32, device=dev) for k in ("valence", "arousal")}
        self.spread = {k: torch.tensor(cal[k], dtype=torch.float32, device=dev) for k in ("valence", "arousal")}
        self.va = (0.0, 0.0)
        for L, ix in json.load(open(a.off)).items():
            idx = torch.tensor(sorted(ix), device=dev)
            def kill(mod, args, idx=idx):
                x = args[0].clone()
                x[..., idx] = 0
                return (x,) + tuple(args[1:])
            self.layers[int(L)].mlp.down_proj.register_forward_pre_hook(kill)
        for r in DRIP:
            def drip(mod, args, out, r=r):
                v, ar = self.va
                if not (v or ar):
                    return out
                add = (v * self.spread["valence"][r] * self.dial["valence"][r] +
                       ar * self.spread["arousal"][r] * self.dial["arousal"][r])
                if isinstance(out, tuple):
                    return (out[0] + add.to(out[0].dtype),) + tuple(out[1:])
                return out + add.to(out.dtype)
            self.layers[r].register_forward_hook(drip)


def choice_logprobs(model, tok, brain, row):
    """log-odds of each choice as his whole answer after his face, the way the brainstem scores them (left-padded, so
    only the last few positions' word odds are computed: the full vocabulary at every position would not fit)"""
    sep = " " if row["face"] else ""
    head = tok(row["prompt"] + row["face"], add_special_tokens=False)["input_ids"]
    tails = [tok(sep + c + "<turn|>", add_special_tokens=False)["input_ids"] for c in row["choices"]]
    n = max(len(t) for t in tails)
    L = len(head) + n
    ids = torch.full((len(tails), L), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(tails), L), dtype=torch.long)
    for i, t in enumerate(tails):
        seq = head + t
        ids[i, L - len(seq):] = torch.tensor(seq)
        mask[i, L - len(seq):] = 1
    brain.va = (row["v"], row["a"])
    pos = (mask.cumsum(1) - 1).clamp(min=0)                            # padding must not shift where words sit
    out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), position_ids=pos.cuda(), logits_to_keep=n + 1)
    lp = torch.log_softmax(out.logits.float(), -1)                     # [choices, n+1, vocab]
    res = []
    for i, t in enumerate(tails):
        k = len(t)
        pos = torch.arange(n + 1 - k - 1, n, device=lp.device)         # the position before each choice word
        res.append(lp[i, pos, torch.tensor(t, device=lp.device)].sum())
    return torch.stack(res)


def anchor_kl(model, tok, brain, row):
    """how far his odds for each word of something he said have moved from his untrained self"""
    p = tok(row["prompt"], add_special_tokens=False)["input_ids"]
    t = tok(row["text"], add_special_tokens=False)["input_ids"][:80]
    ids = torch.tensor([p + t]).cuda()
    brain.va = (row["v"], row["a"])
    k = len(t) + 1
    with torch.no_grad(), model.disable_adapter():
        ref = torch.log_softmax(model(input_ids=ids, logits_to_keep=k).logits.float()[0, :-1], -1)
    cur = torch.log_softmax(model(input_ids=ids, logits_to_keep=k).logits.float()[0, :-1], -1)
    return (cur.exp() * (cur - ref)).sum(-1).mean()


def rows_of(a):
    rows = [json.loads(l) for l in open(a.data)]
    hold = set((a.holdout or "").split(",")) - {""}
    tr = [r for r in rows if r.get("day") not in hold]
    ho = [r for r in rows if r.get("day") in hold]
    return tr, ho


def expected(model, tok, brain, rows, adapter=True):
    """expected score of his odds now, per kind, and the odds themselves"""
    out = {"gate": [], "act": []}
    model.eval()
    for r in rows:
        if r["kind"] not in out:
            continue
        with torch.no_grad():
            if adapter:
                lp = choice_logprobs(model, tok, brain, r)
            else:
                with model.disable_adapter():
                    lp = choice_logprobs(model, tok, brain, r)
        p = torch.softmax(lp, 0).cpu().numpy()
        rw = np.array([r["rewards"][c] for c in r["choices"]])
        out[r["kind"]].append({"t": r["t"], "day": r.get("day"), "p": dict(zip(r["choices"], p.round(4).tolist())),
                               "J": float(p @ rw), "left_min": r.get("left_min"), "alarm": r.get("alarm")})
    return out


def summary(ev):
    s = {}
    for k, xs in ev.items():
        if xs:
            s[k] = {"n": len(xs), "J": round(sum(x["J"] for x in xs) / len(xs), 4)}
    acts = ev.get("act", [])
    for name, lo, hi in (("on time", -20, 10), ("10-25 early", 10, 25), ("25-75 early", 25, 75), ("hours early", 75, 1e9)):
        xs = [x["p"]["now"] for x in acts if x["left_min"] is not None and lo <= x["left_min"] < hi]
        if xs:
            s[f"p(now) {name}"] = round(sum(xs) / len(xs), 3)
    gates = ev.get("gate", [])
    if gates:
        s["p(task) at quiet moments"] = round(sum(1 - x["p"].get("wander", 0) - x["p"].get("no", 0) for x in gates) / len(gates), 3)
    return s


# Immy (2026-10-19): a bad round is thrown away and the next starts from the last accepted model, not built on the
# mistake; only a deep regression is thrown away outright, a mild one we look at together. Deep = any of these
# moves the wrong way by more than this (the pilot: p(now) on time 0.50 -> 0.14). The persona check on the Mac comes after.
DEEP = {"act J": -0.05, "gate J": -0.02, "p(now) on time": -0.15, "p(now) 10-25 early": 0.15, "p(now) 25-75 early": 0.15,
        "p(now) hours early": 0.15}


def verdict(before, after):
    get = lambda s, k: s.get(k.split()[0], {}).get("J") if k.endswith(" J") else s.get(k)
    worse, deep = {}, {}
    for k, lim in DEEP.items():
        b, x = get(before, k), get(after, k)
        if b is None or x is None:
            continue
        d = x - b
        if (lim < 0 and d < 0) or (lim > 0 and d > 0):
            worse[k] = round(d, 4)
            if (lim < 0 and d < lim) or (lim > 0 and d > lim):
                deep[k] = round(d, 4)
    return {"verdict": "deep regression: fall back" if deep else "mild regression: look together" if worse else
            "better or same", "worse": worse, "deep": deep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "train", "eval"])
    ap.add_argument("out")
    for k, d in (("model", None), ("data", None), ("dials", None), ("cal", None), ("off", None), ("holdout", ""),
                 ("adapter", None), ("init", None)):
        ap.add_argument("--" + k, default=d)
    for k, d in (("epochs", 2), ("rank", 16), ("accum", 8), ("seed", 0)):
        ap.add_argument("--" + k, type=int, default=d)
    for k, d in (("lr", 1e-5), ("kl", 2.0), ("anchor", 0.2), ("scale", 10.0)):
        ap.add_argument("--" + k, type=float, default=d)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    tok, model = load(a)
    tr, ho = rows_of(a)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    say = lambda **kw: (print(json.dumps(kw), flush=True), log.write(json.dumps(kw) + "\n"), log.flush())

    if a.mode == "check":
        brain = Brain(model, a)
        diffs, flips = [], 0
        for r in [x for x in tr + ho if x["kind"] in ("gate", "act")][:200]:
            with torch.no_grad():
                p = torch.softmax(choice_logprobs(model, tok, brain, r), 0).cpu().numpy()
            then = np.array([r["logprobs_then"][c] for c in r["choices"]])
            q = np.exp(then - then.max()); q /= q.sum()
            diffs.append(float(np.abs(p - q).max()))
            flips += int(p.argmax() != q.argmax())
        say(mode="check", n=len(diffs), mean_max_prob_diff=round(sum(diffs) / len(diffs), 4),
            worst=round(max(diffs), 4), top_choice_differs=flips)
        return

    from peft import LoraConfig, get_peft_model, PeftModel
    if a.mode == "eval" and a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    elif a.mode == "train" and a.init:                 # build on the last accepted round, never on a rejected one
        model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, bias="none",
                                                 target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj)"))
    brain = Brain(model, a)                    # hooks after LoRA wrapping, so the drip and the off switch see both

    if a.mode == "eval":
        before, after = expected(model, tok, brain, ho, adapter=False), expected(model, tok, brain, ho)
        json.dump({"before": before, "after": after}, open(os.path.join(a.out, "eval.json"), "w"))
        say(mode="eval", before=summary(before), after=summary(after))
        return

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    dec = [r for r in tr if r["kind"] in ("gate", "act")]
    # balance (the pilot, 2026-10-19: most chances to act come early, where now is wrong, so training learned "later"
    # everywhere and his on-time odds of now fell from 0.50 to 0.14; the research's mode collapse): the on-time
    # moments weigh as much in all as the early ones
    group = lambda r: "on" if r["left_min"] is not None and r["left_min"] <= 10 else "early"
    acts = [r for r in dec if r["kind"] == "act"]
    cnt = {g: sum(group(r) == g for r in acts) for g in ("on", "early")}
    w = [len(acts) / (2 * cnt[group(r)]) if r["kind"] == "act" and cnt[group(r)] else 1.0 for r in dec]
    text = [r for r in tr if r["kind"] == "text"]
    ref = {}
    model.eval()
    for i, r in enumerate(dec):                                        # his untrained odds, for the KL guard rail
        with torch.no_grad(), model.disable_adapter():
            ref[i] = torch.log_softmax(choice_logprobs(model, tok, brain, r), 0).detach()
    before = summary(expected(model, tok, brain, ho))   # the accepted model (the starting adapter, or his untrained self)
    say(mode="train", parent=a.init or "untrained", decisions=len(dec), acts_on_time_vs_early=cnt, text=len(text),
        holdout=len(ho), before=before)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        order = list(range(len(dec)))
        random.shuffle(order)
        model.train()
        acc = {"J": 0.0, "kl": 0.0, "anchor": 0.0, "n": 0}
        for j, i in enumerate(order):
            r = dec[i]
            lp = torch.log_softmax(choice_logprobs(model, tok, brain, r), 0)
            p = lp.exp()
            rw = torch.tensor([r["rewards"][c] for c in r["choices"]], device=p.device)
            J = (p * rw).sum()
            kl = (p * (lp - ref[i])).sum()
            loss = w[i] * (-a.scale * J + a.kl * kl) / a.accum
            if text and j % 4 == 0:
                an = anchor_kl(model, tok, brain, random.choice(text))
                loss = loss + a.anchor * an / a.accum
                acc["anchor"] += float(an.detach())
            loss.backward()
            acc["J"] += float(J); acc["kl"] += float(kl); acc["n"] += 1
            if (j + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    n = acc["n"]
                    say(epoch=ep, step=step, J=round(acc["J"] / n, 4), kl=round(acc["kl"] / n, 5),
                        anchor=round(acc["anchor"] / max(1, n // 4), 5), min=round((time.time() - t0) / 60, 1))
                    acc = {"J": 0.0, "kl": 0.0, "anchor": 0.0, "n": 0}
        model.save_pretrained(os.path.join(a.out, "adapter"))
        after = summary(expected(model, tok, brain, ho))
        v = verdict(before, after)
        json.dump({"parent": a.init or "untrained", "before": before, "after": after, **v},
                  open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
        say(epoch_done=ep, holdout=after, **v)


if __name__ == "__main__":
    main()
