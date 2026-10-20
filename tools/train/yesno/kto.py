#!/usr/bin/env python3
"""Training on Immy's yes or no (Immy, 2026-10-19: "The judge will be me. I will go through every single response and
say yes or no"): KTO (Ethayarajh et al. 2024), which learns straight from a thumbs up or down on single replies, no
pairs needed. Each judged reply's odds are raised (yes) or lowered (no) relative to the model it started from, measured
against how far he has moved on replies in general (mismatched prompt and reply pairs), so he can't "win" by drifting
everywhere at once.

  rows   OUT.jsonl --labels L.jsonl [L2.jsonl ...] --items I.jsonl [I2.jsonl ...] --days DIR      (on the Mac)
         each judged item back to the brain call that wrote it (items carry day + step): his exact prompt (his brain's
         own chat template; when he thought first, his private thought too, as context only), his feelings, and his
         raw words. Lines starting "TASK:" (a reminder he set) are never trained: a no for his tone must not teach him
         to stop setting reminders. A reply that ran into its word limit gets no end-of-turn, so a yes never teaches
         him to stop mid-sentence.
  train  OUT --data rows.jsonl --init ACCEPTED_ADAPTER [--epochs 3] [--lr 5e-6] [--beta 0.1] [--group 8]  (on a GPU)
         starts from the last accepted round (Immy's rule) and is measured against it; a fifth of the judged replies
         (yes and no alike) are held out. Ends with OUT/verdict.json: on held-out replies he never trained on, did his
         yes replies get likelier and his no replies less likely, and how far did his words move overall.
common: --model HF_DIR --dials dials.npz --cal calibration.json --off neurons_off.json (as tools/train/choices.py)

Settings follow the TRL KTO guide: beta 0.1, learning rate 5e-7 to 5e-6 (small data: more passes, not a higher rate),
yes and no weighted so neither side outweighs the other by more than 4:3, and at least 15 of each before training.
The decisions he learned in round 2 are checked separately afterwards (choices.py eval on its held-out days), and his
behavior with the 20-sample test (tests/behavior/persona_ab.py); the fallback rule applies to all three.
"""
import argparse, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
END = "<turn|>"                                   # his end of turn (as in choices.py)
THOUGHT_OPEN, THOUGHT_CLOSE = "<|channel>thought\n", "<channel|>"


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def build_rows(a):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "mind"))
    from brain import chat_template
    render = chat_template(os.path.expanduser(os.environ.get("TEMPLATE", "~/.littleguy/brain/brain.gguf")))
    latest = {}
    for f in a.labels:
        for x in jl(f):
            if x["id"] not in latest or x["t"] >= latest[x["id"]]["t"]:
                latest[x["id"]] = x
    items = {x["id"]: x for f in a.items for x in jl(f)}
    steps, rows, missing = {}, [], []
    for i, lab in sorted(latest.items()):
        it = items.get(i)
        if it is None or it.get("step") is None or lab.get("label") not in ("yes", "no"):
            missing.append(i)
            continue
        if it["day"] not in steps:
            steps[it["day"]] = jl(os.path.join(a.days, it["day"], "steps.jsonl"))
        s = steps[it["day"]][it["step"]]
        req, out = s["req"], s["out"]
        thinking = bool(req.get("thinking"))
        prompt = render(req["messages"], thinking)
        if thinking:                                   # what he thought first is context, not something to train
            prompt += THOUGHT_OPEN + out.get("thought", "") + THOUGHT_CLOSE
        lines = out["text"].split("\n")
        pieces = [[l + ("\n" if k < len(lines) - 1 else ""), not l.lstrip().startswith("TASK:")] for k, l in enumerate(lines)]
        cut = out.get("n_tok", 0) >= req.get("max_tokens", 1e9)
        rows.append({"id": i, "label": lab["label"], "note": lab.get("note", ""), "kind": it["kind"], "day": it["day"],
                     "prompt": prompt, "pieces": pieces, "end": "" if cut else END,
                     "v": float(req.get("v", 0)), "a": float(req.get("a", 0))})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "yes": sum(r["label"] == "yes" for r in rows),
                      "no": sum(r["label"] == "no" for r in rows), "notes": sum(bool(r["note"]) for r in rows),
                      "thought_first": sum(THOUGHT_CLOSE in r["prompt"] for r in rows),
                      "cut": sum(not r["end"] for r in rows), "not_found": missing}))


def train(a):
    import torch
    sys.path.insert(0, os.path.join(HERE, ".."))
    from choices import Brain, load
    from peft import PeftModel
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    say = lambda **kw: (print(json.dumps(kw), flush=True), log.write(json.dumps(kw) + "\n"), log.flush())

    rows = jl(a.data)
    yes, no = [r for r in rows if r["label"] == "yes"], [r for r in rows if r["label"] == "no"]
    if min(len(yes), len(no)) < 15:
        raise SystemExit(f"not enough to train on yet: {len(yes)} yes, {len(no)} no (at least 15 of each)")
    hold = set()
    for side in (yes, no):                                # a fifth of each side, never trained on
        hold |= {r["id"] for r in random.sample(side, max(3, len(side) // 5))}
    tr, ho = [r for r in rows if r["id"] not in hold], [r for r in rows if r["id"] in hold]
    n_yes, n_no = sum(r["label"] == "yes" for r in tr), sum(r["label"] == "no" for r in tr)
    w = {"yes": max(1.0, n_no / n_yes), "no": max(1.0, n_yes / n_no)}   # the rarer side counts as much as the other

    tok, model = load(a)
    model = PeftModel.from_pretrained(model, a.init, is_trainable=True)   # the last accepted round
    brain = Brain(model, a)                     # hooks after the adapter, so the drip and the off switch see both

    def logp(prompt_row, words_row, grad):
        """his odds (summed log-odds) for words_row's words after prompt_row's prompt, with prompt_row's feelings"""
        p = tok(prompt_row["prompt"], add_special_tokens=False)["input_ids"]
        ids, mask = [], []
        for text, trained in words_row["pieces"] + ([[words_row["end"], True]] if words_row["end"] else []):
            t = tok(text, add_special_tokens=False)["input_ids"]
            ids += t; mask += [trained] * len(t)
        p = p[-(a.max_prompt):]                                       # the most recent context if a prompt is huge
        x = torch.tensor([p + ids]).cuda()
        brain.va = (prompt_row["v"], prompt_row["a"])
        with torch.set_grad_enabled(grad):
            lg = model(input_ids=x, logits_to_keep=len(ids) + 1).logits[0, :-1].float()
            lp = torch.log_softmax(lg, -1).gather(1, torch.tensor(ids, device=lg.device)[:, None])[:, 0]
            return (lp * torch.tensor(mask, device=lp.device, dtype=lp.dtype)).sum(), sum(mask)

    other = {}                                  # each reply's fixed stranger: someone else's prompt, for the baseline
    for group in (tr, ho):
        ids = [r["id"] for r in group]
        shuffled = ids[1:] + ids[:1]
        random.shuffle(shuffled)
        for i, j in zip(ids, shuffled):
            other[i] = j if j != i else ids[(ids.index(i) + 1) % len(ids)]
    by = {r["id"]: r for r in rows}

    model.eval()
    ref, ref_x, n_words = {}, {}, {}
    for r in rows:                               # the accepted model's odds: fixed, measured once
        with torch.no_grad():
            lp, n = logp(r, r, False); ref[r["id"]] = float(lp); n_words[r["id"]] = n
            ref_x[r["id"]] = float(logp(by[other[r["id"]]], r, False)[0])

    def held_out():
        """on replies he never trained on: how far each moved (log-odds, per reply and per word)"""
        model.eval()
        out = {"yes": [], "no": []}
        with torch.no_grad():
            for r in ho:
                out[r["label"]].append(float(logp(r, r, False)[0]) - ref[r["id"]])
        words = sum(n_words[r["id"]] for r in ho)
        mean = lambda v: round(sum(v) / max(1, len(v)), 4)
        return {"yes_moved": mean(out["yes"]), "no_moved": mean(out["no"]), "n_yes": len(out["yes"]),
                "n_no": len(out["no"]), "yes_up": sum(x > 0 for x in out["yes"]), "no_down": sum(x < 0 for x in out["no"]),
                "per_word_drift": round(sum(abs(x) for v in out.values() for x in v) / max(1, words), 5)}

    before = held_out()
    say(mode="train", parent=a.init, train=len(tr), held_out=len(ho), yes=n_yes, no=n_no, weights=w,
        words_trained=sum(n_words[r["id"]] for r in tr), before=before)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sig = torch.nn.functional.logsigmoid
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        order = tr[:]
        random.shuffle(order)
        for g in range(0, len(order), a.group):
            batch = order[g:g + a.group]
            if len(batch) < 4:                      # the baseline needs a few strangers to mean anything
                continue
            model.eval()
            with torch.no_grad():                   # how far he has moved on replies in general (never below 0)
                z = sum(float(logp(by[other[r["id"]]], r, False)[0]) - ref_x[r["id"]] for r in batch) / len(batch)
            z = max(0.0, z)
            model.train()
            stats = {"loss": 0.0, "r_yes": [], "r_no": []}
            for r in batch:
                lp, _ = logp(r, r, True)
                moved = lp - ref[r["id"]]
                if r["label"] == "yes":
                    loss = w["yes"] * (1 - torch.exp(sig(a.beta * (moved - z))))
                else:
                    loss = w["no"] * (1 - torch.exp(sig(a.beta * (z - moved))))
                (loss / len(batch)).backward()
                stats["loss"] += float(loss) / len(batch)
                stats["r_" + r["label"]].append(float(moved))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); step += 1
            mean = lambda v: round(sum(v) / len(v), 3) if v else None
            say(epoch=ep, step=step, loss=round(stats["loss"], 4), baseline=round(z, 3), yes_moved=mean(stats["r_yes"]),
                no_moved=mean(stats["r_no"]), min=round((time.time() - t0) / 60, 1))
        model.save_pretrained(os.path.join(a.out, "adapter"))
        after = held_out()
        sep_before, sep_after = before["yes_moved"] - before["no_moved"], after["yes_moved"] - after["no_moved"]
        v = {"parent": a.init, "before": before, "after": after, "separation": round(sep_after - sep_before, 4),
             "learned": after["yes_moved"] > after["no_moved"] and after["no_moved"] < 0,
             "next": "check decisions (choices.py eval) and behavior (persona_ab.py) before any verdict"}
        json.dump(v, open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
        say(epoch_done=ep, held_out=after, separation=v["separation"], learned=v["learned"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rows", "train"])
    ap.add_argument("out")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--items", nargs="*", default=[])
    for k in ("days", "data", "init", "model", "dials", "cal", "off"):
        ap.add_argument("--" + k)
    for k, d in (("epochs", 3), ("group", 8), ("seed", 0), ("max_prompt", 3072)):
        ap.add_argument("--" + k, type=int, default=d)
    for k, d in (("lr", 5e-6), ("beta", 0.1)):
        ap.add_argument("--" + k, type=float, default=d)
    a = ap.parse_args()
    build_rows(a) if a.mode == "rows" else train(a)


if __name__ == "__main__":
    main()
