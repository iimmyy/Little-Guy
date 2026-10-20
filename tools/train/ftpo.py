#!/usr/bin/env python3
"""Taking out what still breaks Immy's rules, word by word: Final Token Preference Optimization (FTPO, Antislop, Paech
et al. 2025, arXiv 2510.15061; the loss below is their trainer's, github.com/sam-paech/auto-antislop, MIT). Where he
starts something Immy's rules forbid (an exclamation mark, an emoji, a list or bold, a robot or banned word, a
distill.py's rules, written from Immy's no's; the "[thought]" scripts are about how a whole line is
built, so they stay with distill.py), the first word of it is "rejected" and the other
words he himself found likely there are "chosen". Only that one next word is trained, with the rest of his word odds
tied to where they were, so nothing else about him moves. No AI judges anything: the rules are Immy's, and the chosen
words are his own.

  pairs  OUT.jsonl --days DIR_OR_JSONL... --init ADAPTER [--min_p 0.05]   (GPU) lines this same brain wrote with his
         live prompts (no sheet; practice days or replay.py): where the first forbidden thing starts in a line, that
         spot is a pair (the paper makes them while sampling with backtracking; one pair per finished line is the same
         spot without the sampler)
  train  OUT --data pairs.jsonl --init ADAPTER [--rank 16] [--lr 2e-5] [--layers 10]   (GPU) a new adapter on top of
         the accepted one, only in his last LAYERS rooms (the paper's recipe), stopped once 85% of chosen words win
common: --model HF_DIR --dials --cal --off (as choices.py). Join with distill.py join (missing rooms are zeros).
"""
import argparse, json, os, random, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from distill import ROBOT, SURE, LISTY, FACE

EXCLAIM = re.compile(r"!")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def first_break(text):
    """where the first forbidden thing starts in text (character offset), and what it is; None if nothing"""
    hits = []
    for name, rx in (("exclamation mark", EXCLAIM), ("emoji", EMOJI), ("list or bold", LISTY), ("banned word", ROBOT),
                     ("banned word", SURE)):
        for m in rx.finditer(text):
            if name == "banned word" and any(f.start() <= m.start() < f.end() for f in FACE.finditer(text)):
                continue                                      # a face like [ processing ] is a face, not a word
            start = m.start() + (len(m.group(0)) - len(m.group(0).lstrip(" .?,\n")) if name in ("banned word", "list or bold") else 0)
            hits.append((start, name))
            break
    return min(hits) if hits else None


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def setup(a, trainable):
    import torch
    from choices import Brain, load
    from peft import PeftModel, LoraConfig, get_peft_model
    tok, model = load(a)
    model = PeftModel.from_pretrained(model, a.init).merge_and_unload()
    if trainable:
        n = 42
        rooms = "|".join(str(i) for i in range(n - a.layers, n))
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, bias="none",
                                                 target_modules=rf".*language_model.*layers\.({rooms})\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj)"))
    return tok, model, Brain(model, a)


def pairs(a):
    """his own finished lines (steps.jsonl from days or a replay, written by the brain being trained): at the first
    spot in each where something forbidden starts, the word he wrote is rejected and the allowed words he found likely
    right there are chosen (one look at that spot, so thousands of lines take minutes)"""
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "mind"))
    from brain import chat_template
    render = chat_template(a.template)
    random.seed(a.seed); torch.manual_seed(a.seed)
    tok, model, brain = setup(a, False)
    model.eval()
    end_ids = {tok.convert_tokens_to_ids("<turn|>"), tok.eos_token_id}
    out, seen, made, kinds, t0 = open(a.out, "w"), 0, 0, {}, time.time()
    for d in a.days:
        f = d if d.endswith(".jsonl") else os.path.join(d, "steps.jsonl")
        for s in jl(f):
            text = s.get("out", {}).get("text") if s.get("op", "think") == "think" else None
            if not text:
                continue
            seen += 1
            br = first_break(text)
            if br is None:
                continue
            req = s["req"]
            prompt = render(req["messages"], bool(req.get("thinking")))
            if req.get("thinking"):
                prompt += "<|channel>thought\n" + s["out"].get("thought", "") + "<channel|>"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"][-a.max_prompt:]
            t_ids = tok(text, add_special_tokens=False)["input_ids"]
            i = next(i for i in range(len(t_ids)) if len(tok.decode(t_ids[:i + 1])) > br[0])
            gen, rejected = t_ids[:i], t_ids[i]
            brain.va = (float(req.get("v", 0)), float(req.get("a", 0)))
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([p_ids + gen]).cuda(), logits_to_keep=1).logits[0, -1].float()
            p = torch.softmax(logits, -1)
            cand = torch.nonzero(p >= a.min_p * p.max()).flatten().tolist()
            chosen = [c for c in sorted(cand, key=lambda c: -float(p[c])) if c != rejected and c not in end_ids
                      and first_break(tok.decode(gen + [c])) is None][:a.max_chosen]
            if len(chosen) < a.min_chosen:
                continue
            out.write(json.dumps({"prompt_ids": p_ids + gen, "rejected": rejected, "chosen": chosen, "what": br[1],
                                  "context": tok.decode(gen)[-80:], "rejected_text": tok.decode([rejected]),
                                  "p_rejected": round(float(p[rejected]), 4),
                                  "chosen_text": [tok.decode([c]) for c in chosen[:6]],
                                  "v": brain.va[0], "a": brain.va[1]}, ensure_ascii=False) + "\n")
            made += 1
            kinds[br[1]] = kinds.get(br[1], 0) + 1
            if made % 50 == 0:
                print(json.dumps({"lines": seen, "pairs": made, "min": round((time.time() - t0) / 60, 1)}), flush=True)
    print(json.dumps({"lines": seen, "pairs": made, "kinds": kinds}))


def train(a):
    import torch
    import torch.nn.functional as F
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    say = lambda **kw: (print(json.dumps(kw), flush=True), log.write(json.dumps(kw) + "\n"), log.flush())
    rows = jl(a.data)
    # the paper's sample regularisation: one kind of break (say "!") must not be most of the data
    by = {}
    for r in rows:
        by.setdefault(r["rejected"], []).append(r)
    cap = max(20, int(len(rows) * 0.15))
    rows = [x for v in by.values() for x in (random.sample(v, cap) if len(v) > cap else v)]
    random.shuffle(rows)
    ho, tr = rows[:max(20, len(rows) // 10)], rows[max(20, len(rows) // 10):]
    tok, model, brain = setup(a, True)

    def last_logits(r, adapter=True):
        x = torch.tensor([r["prompt_ids"][-a.max_prompt:]]).cuda()
        brain.va = (r["v"], r["a"])
        if adapter:
            return model(input_ids=x, logits_to_keep=1).logits[0, -1].float()
        with torch.no_grad(), model.disable_adapter():
            return model(input_ids=x, logits_to_keep=1).logits[0, -1].float()

    def loss_of(r):
        """FTPO (auto-antislop core/ftpo_trainer.py compute_loss, their defaults): chosen words must beat the rejected
        one by a margin of 2 logits, after which they stop pulling; the rejected and chosen logits may move up to 0.5
        freely, beyond that lightly tied (0.05); every other word's logit is tied to where it was (0.4)"""
        y, y0 = last_logits(r), last_logits(r, False)
        ch = torch.tensor(r["chosen"], device=y.device)
        delta = y[ch] - y[r["rejected"]]
        w = torch.clamp((a.margin - delta) / a.margin, 0, 1)
        pref = (F.softplus(a.margin - delta) * w).mean()
        tgt = torch.zeros_like(y, dtype=torch.bool)
        tgt[ch] = True; tgt[r["rejected"]] = True
        diff = y - y0
        mse_rest = diff[~tgt].pow(2).mean()
        mse_tgt = torch.clamp(diff[tgt].abs() - a.tau, min=0).pow(2).mean()
        win = float((y[ch] > y[r["rejected"]]).float().mean())
        return pref + a.lam_rest * mse_rest + a.lam_tgt * mse_tgt, win

    def held_out():
        model.eval()
        with torch.no_grad():
            wins = [loss_of(r)[1] for r in ho]
        return round(sum(wins) / len(wins), 4)

    before = held_out()
    kinds = {}
    for r in tr:
        kinds[r["what"]] = kinds.get(r["what"], 0) + 1
    say(mode="train", parent=a.init, train=len(tr), held_out=len(ho), kinds=kinds, chosen_win_before=before)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    step, t0, acc = 0, time.time(), {"loss": 0.0, "win": 0.0, "n": 0}
    done = False
    for ep in range(a.epochs):
        model.train()
        random.shuffle(tr)
        for j, r in enumerate(tr):
            loss, win = loss_of(r)
            (loss / a.accum).backward()
            acc["loss"] += float(loss.detach()); acc["win"] += win; acc["n"] += 1
            if (j + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    win_now = acc["win"] / acc["n"]
                    say(epoch=ep, step=step, loss=round(acc["loss"] / acc["n"], 4), chosen_win=round(win_now, 3),
                        min=round((time.time() - t0) / 60, 1))
                    acc = {"loss": 0.0, "win": 0.0, "n": 0}
                    if win_now >= a.stop_win:                   # the paper's early stop: don't overtrain
                        done = True
                        break
        if done:
            break
    model.save_pretrained(os.path.join(a.out, "adapter"))
    after = held_out()
    json.dump({"parent": a.init, "chosen_win_before": before, "chosen_win_after": after, "steps": step},
              open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
    say(done=True, chosen_win_held_out=after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pairs", "train"])
    ap.add_argument("out")
    ap.add_argument("--days", nargs="*", default=[])
    for k in ("data", "init", "model", "dials", "cal", "off"):
        ap.add_argument("--" + k)
    ap.add_argument("--template", default=os.path.expanduser("~/work/lg/brain/brain.gguf"))
    for k, d in (("per", 4), ("max_prompts", 1500), ("max_words", 120), ("max_prompt", 3072), ("min_chosen", 2),
                 ("max_chosen", 12), ("rank", 16), ("layers", 10), ("accum", 16), ("epochs", 2), ("seed", 0)):
        ap.add_argument("--" + k, type=int, default=d)
    for k, d in (("min_p", 0.05), ("lr", 2e-5), ("margin", 2.0), ("tau", 0.5), ("lam_rest", 0.4), ("lam_tgt", 0.05),
                 ("stop_win", 0.85)):
        ap.add_argument("--" + k, type=float, default=d)
    a = ap.parse_args()
    pairs(a) if a.mode == "pairs" else train(a)


if __name__ == "__main__":
    main()
