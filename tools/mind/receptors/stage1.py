#!/usr/bin/env python3
"""Receptors inside his brain, stage 1 (spec: .run/results/mind/RECEPTORS-IN-THE-MODEL-spec-v1.md). Each of the 42
rooms gets three multipliers, set per request: focus (attention sharpness), excitability (the neurons' firing-curve
gain) and loudness (how strongly the room's result joins the stream). All 1.0 is today's brain.

  identity <out.json>     today's brain vs the receptors all at 1.0: the same words and the same odds, exactly
  sweep <out.json> <term> <m1> [<m2> ...]   one term in every room at each multiplier: his replies, read by us

env LG_MIND_RECEP: the llama-lg-mind built against the receptor engine; LG_BRAIN, LG_LORA: the brain and its adapter
(default: the live ones in ~/.littleguy/brain). The brain runs greedy so a change in words is the receptors' doing.
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from brain import chat_template

HOME = os.path.expanduser("~")
BIN = os.environ.get("LG_MIND_RECEP", ".run/results/mind/receptors-stage1/bin/llama-lg-mind")
BRAIN = os.environ.get("LG_BRAIN", f"{HOME}/.littleguy/brain/brain.gguf")
LORA = os.environ.get("LG_LORA", f"{HOME}/.littleguy/brain/adapter.gguf")
TERMS = {"focus": 0, "excite": 1, "loud": 2}
PROMPTS = ["Hey.", "How's it going?", "Tell me about octopuses.", "Plan me a quick dinner with eggs and rice.",
           "I can't sleep and my heart is racing.", "My friend didn't text back all day."]
CHOICES = ["yes", "no", "maybe", "I don't know"]


_render = chat_template(BRAIN)


def render(user):
    return _render([{"role": "user", "content": user}])


class Mind:
    def __init__(self):
        args = [BIN, "-m", BRAIN, "-ngl", "99", "-c", "4096"] + (["--lora", LORA] if LORA else [])
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1)
        ready = json.loads(self.p.stdout.readline())
        self.n_layer = ready["n_layer"]

    def ask(self, req):
        self.p.stdin.write(json.dumps(req) + "\n")
        out = json.loads(self.p.stdout.readline())
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def reply(self, user, recep, n=80):
        return self.ask({"op": "think", "prompt": render(user), "cvec": None, "recep": recep, "max_tokens": n,
                         "temperature": 0.0, "top_p": 1.0, "seed": 0})["text"]

    def odds(self, user, recep):
        return self.ask({"op": "think", "prompt": render(user), "cvec": None, "recep": recep,
                         "choices": [c + "<turn|>" for c in CHOICES]})["choice_logprobs"]

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def grid(n_layer, term=None, m=1.0):
    g = [1.0] * (3 * n_layer)
    if term is not None:
        for il in range(n_layer):
            g[3 * il + TERMS[term]] = m
    return g


def identity(out):
    mind = Mind()
    ones = grid(mind.n_layer)
    rows = []
    for u in PROMPTS:
        a, b = mind.reply(u, None), mind.reply(u, ones)
        oa, ob = mind.odds(u, None), mind.odds(u, ones)
        rows.append({"prompt": u, "same_words": a == b, "same_odds": oa == ob, "today": a, "ones": b,
                     "odds_today": oa, "odds_ones": ob})
        print(f"{'SAME' if a == b and oa == ob else 'DIFF'}  {u}", flush=True)
    mind.close()
    json.dump({"brain": BRAIN, "lora": LORA, "rows": rows}, open(out, "w"), indent=1)
    print(f"{sum(r['same_words'] and r['same_odds'] for r in rows)} of {len(rows)} identical")


def sweep(out, term, mults):
    mind = Mind()
    rows = []
    for u in PROMPTS:
        for m in mults:
            text = mind.reply(u, grid(mind.n_layer, term, m))
            rows.append({"prompt": u, "term": term, "mult": m, "text": text})
            print(f"--- {term} x{m}  {u}\n{text}\n", flush=True)
    mind.close()
    json.dump({"brain": BRAIN, "lora": LORA, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "identity":
        identity(sys.argv[2])
    elif sys.argv[1] == "sweep":
        sweep(sys.argv[2], sys.argv[3], [float(x) for x in sys.argv[4:]])
