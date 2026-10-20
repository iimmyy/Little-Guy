#!/usr/bin/env python3
"""Teaching him to talk the way the coaching sheet makes him talk, without the sheet (context distillation: Askell et
al. 2021, Snell et al. 2022 "Learning by Distilling Context"). Why: Immy said yes to 9 of 25 coached replies and to 0 of
71 uncoached ones, and live he has no sheet. The practice days ran with the sheet; here the sheet comes out of every
prompt and he is trained to say what he said with it.

No AI decides what is good (Immy, 2026-10-19: "AI cannot grade something on making it sound not AI"). A coached reply
is kept or dropped only by Immy's own rules and answers:
  - every reply Immy said no to is dropped (yes ones are kept)
  - his notes on his first no's, as written on the sheet: no exclamation marks; no sure, awesome, absolutely, definitely,
    certainly, great question; no lists, bold or emojis; no robot words (data, processing, systems, sensors,
    protocols, matrix, logging); no um or uh
  - "Friday already huh is repeated" (b4-0009): a quiet-moment thought that mostly repeats one he already had that day
  - "Script" / "What" (b4-0018, b4-0021, b4-0022, b4-0026): a thought that labels itself [thought] or [thinking] and
    then says a second line
  - replies that ran out of words (Immy can't judge those, so neither are they taught)
What he thought privately before answering (the checklist) is context only, never trained.

  rows   OUT.jsonl --days DIR... --coach coach.txt [--labels L.jsonl...] [--items I.jsonl...] [--holdout day-009]
         (on the Mac: renders his prompts with his brain's own chat template, as kto.py does)
  train  OUT --data rows.jsonl --init ACCEPTED_ADAPTER [--rank 16] [--lr 5e-5] [--kl 0.1] [--epochs 1]  (GPU)
         the accepted round is folded into the weights and a new adapter learns on top, so "how far he moved" is
         always measured against the accepted model (the anchor: per-word KL to it, so only how he talks changes)
  join   OUT_DIR --init ACCEPTED_ADAPTER --new NEW_ADAPTER    one adapter that is both (stacked ranks, exact), for
         the brainstem's single --lora
common: --model HF_DIR --dials --cal --off (as tools/train/choices.py)
"""
import argparse, json, os, random, re, sys, time, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
END = "<turn|>"
THOUGHT_OPEN, THOUGHT_CLOSE = "<|channel>thought\n", "<channel|>"

ROBOT = re.compile(r"\b(data|processing|systems?|sensors?|protocols?|matrix|logging|absolutely|definitely|certainly|"
                   r"awesome|um|uh|great question)\b", re.I)
SURE = re.compile(r"(?:^|[.?,]\s*)sure\b", re.I)     # "Sure, I can..." (not "not sure")
LISTY = re.compile(r"\*\*|^\s*([-*•]|\d+\.)\s", re.M)
SELF_LABEL = re.compile(r"\[\s*(thought|thinking)\b", re.I)
FACE = re.compile(r"\[[^\]\n]*\]")


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def emoji(text):
    return any(unicodedata.category(ch) == "So" for ch in text)


def why_dropped(text, earlier):
    if "!" in text:
        return "exclamation mark"
    if emoji(text):
        return "emoji"
    if LISTY.search(text):
        return "list or bold"
    m = ROBOT.search(FACE.sub(" ", text)) or SURE.search(FACE.sub(" ", text).strip())
    if m:
        return "robot or banned word: " + m.group(0).strip(" .?,").lower()
    if SELF_LABEL.search(text) and len(FACE.findall(text)) > 1:
        return "script ([thought] then a second line)"
    words = set(re.findall(r"[a-z']+", FACE.sub(" ", text).lower()))
    for e in earlier:
        if words and len(words & e) / len(words) >= 0.6:
            return "repeats a thought he already had today"
    return None


def build_rows(a):
    sys.path.insert(0, os.path.join(HERE, "..", "mind"))
    from brain import chat_template
    render = chat_template(os.path.expanduser(os.environ.get("TEMPLATE", "~/.littleguy/brain/brain.gguf")))
    coach = "".join(l for l in open(a.coach) if not l.startswith("#")).strip()
    latest = {}
    for f in a.labels:
        for x in jl(f):
            if x["id"] not in latest or x["t"] >= latest[x["id"]]["t"]:
                latest[x["id"]] = x
    items = {x["id"]: x for f in a.items for x in jl(f)}
    said = {}                                     # (day, step) -> Immy's answer on it
    for i, lab in latest.items():
        it = items.get(i)
        if it and it.get("step") is not None and it.get("part") != "thinking":
            said[(it["day"], it["step"])] = lab["label"]
    hold = set((a.holdout or "").split(",")) - {""}
    rows, dropped = [], {}
    for d in a.days:
        day = os.path.relpath(d, os.path.dirname(os.path.dirname(d)))       # "coached-1/day-003", as items name it
        earlier = []
        for k, s in enumerate(jl(os.path.join(d, "steps.jsonl"))):
            out = s["out"]
            if s["op"] != "think" or not out.get("text"):
                continue
            req = dict(s["req"])
            text = out["text"]
            why = None
            if said.get((day, k)) == "no":
                why = "Immy said no"
            elif out.get("n_tok", 0) >= req.get("max_tokens", 1e9):
                why = "ran out of words"
            elif said.get((day, k)) != "yes":
                why = why_dropped(text, earlier)
            words = set(re.findall(r"[a-z']+", FACE.sub(" ", text).lower()))
            earlier.append(words)
            if why:
                dropped[why.split(":")[0]] = dropped.get(why.split(":")[0], 0) + 1
                continue
            msgs = [dict(m, content=m["content"].replace("\n\n" + coach, "").replace(coach, ""))
                    for m in req["messages"]]
            if msgs == req["messages"]:
                raise SystemExit(f"{day} step {k}: the sheet is not in this prompt (wrong --coach?)")
            thinking = bool(req.get("thinking"))
            prompt = render(msgs, thinking)
            if thinking:
                prompt += THOUGHT_OPEN + out.get("thought", "") + THOUGHT_CLOSE
            rows.append({"day": day, "step": k, "held_out": os.path.basename(d) in hold,
                         "kind": "reply" if "is talking with you" in msgs[0]["content"] else "thought",
                         "immy": said.get((day, k)), "prompt": prompt, "text": text, "end": END,
                         "v": float(req.get("v", 0)), "a": float(req.get("a", 0))})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(json.dumps({"kept": len(rows), "by_kind": kinds, "held_out": sum(r["held_out"] for r in rows),
                      "immy_yes_kept": sum(r["immy"] == "yes" for r in rows), "dropped": dropped}, indent=1))


def train(a):
    import torch
    sys.path.insert(0, HERE)
    from choices import Brain, load
    from peft import PeftModel, LoraConfig, get_peft_model
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    say = lambda **kw: (print(json.dumps(kw), flush=True), log.write(json.dumps(kw) + "\n"), log.flush())
    rows = jl(a.data)
    tr, ho = [r for r in rows if not r["held_out"]], [r for r in rows if r["held_out"]]
    replies = [r for r in tr if r["kind"] == "reply"]
    thoughts = [r for r in tr if r["kind"] == "thought"]
    if len(thoughts) > a.thought_ratio * len(replies):        # his quiet thoughts outnumber replies ~8 to 1; Immy
        thoughts = random.sample(thoughts, int(a.thought_ratio * len(replies)))   # hears the replies
    tr = replies + thoughts

    tok, model = load(a)
    model = PeftModel.from_pretrained(model, a.init).merge_and_unload()     # the accepted round, folded in
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, bias="none",
                                             target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj)"))
    brain = Brain(model, a)

    def ids_of(r):
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"][-a.max_prompt:]
        t = tok(r["text"] + r["end"], add_special_tokens=False)["input_ids"][:a.max_words]
        return p, t

    def forward(r, grad):
        """per-word log-odds of his coached words (student) and the accepted model's, without the sheet"""
        p, t = ids_of(r)
        x = torch.tensor([p + t]).cuda()
        brain.va = (r["v"], r["a"])
        k = len(t) + 1
        tgt = torch.tensor(t, device="cuda")
        with torch.no_grad(), model.disable_adapter():
            ref = torch.log_softmax(model(input_ids=x, logits_to_keep=k).logits[0, :-1].float(), -1)
        with torch.set_grad_enabled(grad):
            cur = torch.log_softmax(model(input_ids=x, logits_to_keep=k).logits[0, :-1].float(), -1)
        nll = -cur.gather(1, tgt[:, None])[:, 0]
        kl = (cur.exp() * (cur - ref)).sum(-1)
        ref_nll = -ref.gather(1, tgt[:, None])[:, 0]
        return nll.mean(), kl.mean(), float(ref_nll.mean())

    def held_out():
        model.eval()
        n, s, s0, d = 0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for r in ho[:a.eval_n]:
                nll, kl, ref_nll = forward(r, False)
                n += 1; s += float(nll); s0 += ref_nll; d += float(kl)
        return {"n": n, "coached_words_nll": round(s / max(1, n), 4), "accepted_model_nll": round(s0 / max(1, n), 4),
                "kl_to_accepted": round(d / max(1, n), 4)}

    before = held_out()
    say(mode="train", parent=a.init, train=len(tr), replies=len(replies), thoughts=len(thoughts), held_out=len(ho),
        before=before)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        order = tr[:]
        random.shuffle(order)
        model.train()
        acc = {"nll": 0.0, "kl": 0.0, "n": 0}
        for j, r in enumerate(order):
            nll, kl, _ = forward(r, True)
            ((nll + a.kl * kl) / a.accum).backward()
            acc["nll"] += float(nll); acc["kl"] += float(kl); acc["n"] += 1
            if (j + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    say(epoch=ep, step=step, nll=round(acc["nll"] / acc["n"], 4), kl=round(acc["kl"] / acc["n"], 4),
                        min=round((time.time() - t0) / 60, 1))
                    acc = {"nll": 0.0, "kl": 0.0, "n": 0}
        model.save_pretrained(os.path.join(a.out, "adapter"))
        after = held_out()
        json.dump({"parent": a.init, "before": before, "after": after}, open(os.path.join(a.out, "verdict.json"), "w"),
                  indent=1)
        say(epoch_done=ep, held_out=after)


def join(a):
    """accepted (rank r1) + new (rank r2) as one adapter of rank r1 + r2: A stacked, B side by side, each B carrying its
    own scale, so the sum of the two changes is exact"""
    import torch
    from safetensors.torch import load_file, save_file
    cfg1, cfg2 = (json.load(open(os.path.join(d, "adapter_config.json"))) for d in (a.init, a.new))
    w1, w2 = (load_file(os.path.join(d, "adapter_model.safetensors")) for d in (a.init, a.new))
    s1, s2 = cfg1["lora_alpha"] / cfg1["r"], cfg2["lora_alpha"] / cfg2["r"]
    out = {}
    for k, A1 in w1.items():
        if ".lora_A." not in k:
            continue
        kb = k.replace(".lora_A.", ".lora_B.")
        A2, B2 = w2.get(k), w2.get(kb)
        if A2 is None:
            raise SystemExit(f"{k} is missing from the new adapter")
        out[k] = torch.cat([A1, A2], 0).contiguous()
        out[kb] = torch.cat([w1[kb] * s1, B2 * s2], 1).contiguous()
    extra = [k for k in w2 if k not in out and k.replace(".lora_B.", ".lora_A.") not in out]
    if extra:
        raise SystemExit(f"the new adapter has modules the accepted one doesn't: {extra[:3]}")
    r = cfg1["r"] + cfg2["r"]
    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "adapter_model.safetensors"))
    json.dump(dict(cfg1, r=r, lora_alpha=r), open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)
    print(json.dumps({"rank": r, "modules": len(out) // 2}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rows", "train", "join"])
    ap.add_argument("out")
    ap.add_argument("--days", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--items", nargs="*", default=[])
    for k in ("data", "init", "new", "model", "dials", "cal", "off", "coach", "holdout"):
        ap.add_argument("--" + k)
    for k, d in (("epochs", 1), ("rank", 16), ("accum", 8), ("seed", 0), ("max_prompt", 3072), ("max_words", 400),
                 ("eval_n", 200)):
        ap.add_argument("--" + k, type=int, default=d)
    for k, d in (("lr", 5e-5), ("kl", 0.1), ("thought_ratio", 2.0)):
        ap.add_argument("--" + k, type=float, default=d)
    a = ap.parse_args()
    {"rows": build_rows, "train": train, "join": join}[a.mode](a)


if __name__ == "__main__":
    main()
