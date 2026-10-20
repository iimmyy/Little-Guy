#!/usr/bin/env python3
"""Little Guy's brainstem: his brain (llama.cpp, engine/lg-mind) with his two core dials, the same math as
life_sim.py's PyTorch hooks, so his mind loop can run on the Mac (and later the Jetson) instead of a cloud GPU.

  dials   <directions.pt> <valence key> <arousal key> <out.npz>   one-time: the dial directions as unit vectors
          (needs torch; everything else needs only numpy + jinja2)
  check   <brain.gguf> <dials.npz> <life.json> <scenario> <way> <out.json>   recompute how his notes from a life sim
          read inside him (rooms 24-36) and compare with the cloud run's own readings

Brain(...) in code:
  calibrate()          natural spread of each dial on his ordinary answers (life_sim.py does this at start)
  think(msgs, v, a)    a reply with his feelings as a drip: v and a natural steps along each dial, rooms 18-40
  think_full(.., thinking=True)   the same with the brain's own thinking switch on: {"text", "thought", "n_tok"}
  choose(msgs, choices, v, a)     the odds of each choice as the whole answer (a decision's policy)
  read_feel(text)      how pleasant / keyed up a text reads inside him, in natural steps (rooms 24-36)
env LG_LORA: an adapter (what training added) loaded on top of his frozen brain, for thinking only
"""
import json, os, re, signal, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LG_MIND = os.environ.get("LG_MIND", os.path.join(HERE, "..", "..", "engine", "lg-mind", "build", "llama-lg-mind"))
DRIP = range(18, 41)                       # rooms the feelings drip into
READ = range(24, 37)                       # rooms his notes are read from
CAL = ["Hey.", "How's it going?", "Tell me about octopuses.", "Plan me a quick dinner with eggs and rice."]


def chat_template(gguf_path):
    """the brain's own chat template (from its GGUF, or a copy saved next to it as <gguf>.jinja), rendered the way
    transformers does"""
    import jinja2
    if os.path.exists(gguf_path + ".jinja"):
        text = open(gguf_path + ".jinja").read()
    else:
        sys.path.insert(0, os.path.join(HERE, "..", "..", ".run", "lcpp-src", "gguf-py"))
        from gguf import GGUFReader
        f = GGUFReader(gguf_path).fields["tokenizer.chat_template"]
        text = bytes(f.parts[f.data[0]]).decode()
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    def fail(m):
        raise ValueError(m)
    env.globals["raise_exception"] = fail
    t = env.from_string(text)
    return lambda msgs, thinking=False: t.render(messages=msgs, add_generation_prompt=True, bos_token="<bos>",
                                                 enable_thinking=thinking)


THOUGHT = re.compile(r"^\s*<\|channel>thought\n?(.*?)(?:<channel\|>(.*))?$", re.S)
SPECIAL = re.compile(r"<\|?[a-z_]+\|?>")


def split_thought(raw):
    """thinking mode writes <|channel>thought ... <channel|> then the answer; no closing marker = he ran out of words
    while still thinking (no answer yet)"""
    m = THOUGHT.match(raw)
    if not m:
        return "", SPECIAL.sub("", raw).strip()
    return m.group(1).strip(), SPECIAL.sub("", m.group(2) or "").strip()


class Brain:
    def __init__(self, gguf, dials_npz, ctx=4096, log=None):
        d = np.load(dials_npz)
        self.dial = {"valence": d["valence"].astype(np.float32), "arousal": d["arousal"].astype(np.float32)}
        self.render = chat_template(gguf)
        self.tmp = tempfile.mkdtemp(prefix="lg-brain-")
        lora = ["--lora", os.environ["LG_LORA"]] if os.environ.get("LG_LORA") else []   # what he learned (choices.py)
        self.p = subprocess.Popen([LG_MIND, "-m", gguf, "-ngl", "99", "-c", str(ctx)] + lora, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL, text=True)
        hello = json.loads(self.p.stdout.readline())
        self.n_embd, self.n_layer = hello["n_embd"], hello["n_layer"]
        self.spread = None
        self._cvec_n = 0

    def _ask(self, req):
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        out = json.loads(self.p.stdout.readline())
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def _states(self, prompt, prefix, rooms, mode):
        path = os.path.join(self.tmp, "states.f32")
        n = self._ask({"op": "states", "prompt": prompt, "prefix": prefix, "rooms": list(rooms), "out": path,
                       "mode": mode})["n_tok"]
        x = np.fromfile(path, dtype=np.float32)
        return x.reshape(len(rooms), self.n_embd) if mode == "mean" else x.reshape(len(rooms), n, self.n_embd)

    def calibrate(self):
        """life_sim.py: greedy 120-word answers to CAL, each dial's projection on those words, per room: spread = std,
        rest = mean (in the same units as mood_dose.py's steps)"""
        proj = {k: [] for k in self.dial}
        for q in CAL:
            prompt = self.render([{"role": "user", "content": q}])
            ans = self._ask({"op": "think", "prompt": prompt, "max_tokens": 120})["text"]
            H = self._states(prompt + ans, prompt, range(self.n_layer), "all")          # rooms x words x n_embd
            for k in self.dial:
                proj[k].append(np.einsum("ltw,lw->lt", H, self.dial[k]))
        self.spread = {}
        for k in self.dial:
            P = np.concatenate(proj[k], axis=1)
            self.spread[k], self.spread[k + "_rest"] = P.std(axis=1, ddof=1), P.mean(axis=1)
        return self.spread

    def _cvec(self, v, a):
        if not (v or a):
            return None
        buf = np.zeros((self.n_layer - 1, self.n_embd), np.float32)            # row r-1 = room r (llama.cpp layout)
        for r in DRIP:
            buf[r - 1] = v * self.spread["valence"][r] * self.dial["valence"][r] + \
                         a * self.spread["arousal"][r] * self.dial["arousal"][r]
        self._cvec_n += 1                                                       # a new file name = a new push
        path = os.path.join(self.tmp, f"cvec{self._cvec_n}.f32")
        buf.tofile(path)
        return path

    def think(self, msgs, v=0.0, a=0.0, max_tokens=200, temperature=0.0, top_p=1.0, seed=0):
        return self._ask({"op": "think", "prompt": self.render(msgs), "cvec": self._cvec(v, a), "max_tokens": max_tokens,
                          "temperature": temperature, "top_p": top_p, "seed": seed})["text"]

    def think_full(self, msgs, v=0.0, a=0.0, max_tokens=200, temperature=0.0, top_p=1.0, seed=0, thinking=False,
                   think_tokens=0):
        """think_tokens (with thinking): his private thought's own budget, closed for him when it runs out, so
        max_tokens is all his answer's. "hushed": he stopped mid-writing because he heard something (see hush)."""
        r = self._ask({"op": "think", "prompt": self.render(msgs, thinking), "cvec": self._cvec(v, a),
                       "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "seed": seed,
                       "special": thinking, "think_tokens": think_tokens if thinking else 0})
        thought, text = split_thought(r["text"]) if thinking else ("", r["text"])
        return {"text": text, "thought": thought, "n_tok": r["n_tok"], "n_thought": r.get("n_thought", 0),
                "hushed": r.get("hushed", False)}

    def hush(self):
        """stop the writing in progress at the next word (lg-mind's SIGUSR1)"""
        self.p.send_signal(signal.SIGUSR1)

    def choose(self, msgs, choices, v=0.0, a=0.0, seed=0):
        """log-odds of each choice as his whole answer (the choice, then the end of his turn). He opens almost every
        answer with a face ("[ focused ]"), which leaves a bare word nearly no chance, so he makes his face first (as
        always, sampled with the drip on) and the choices are scored after it."""
        prompt, cv = self.render(msgs), self._cvec(v, a)
        head = self._ask({"op": "think", "prompt": prompt, "cvec": cv, "max_tokens": 12, "temperature": 0.8,
                          "top_p": 0.95, "seed": seed})["text"]
        face = head[:head.index("]") + 1] if head.lstrip().startswith("[") and "]" in head else ""
        sep = " " if face else ""
        lp = self._ask({"op": "think", "prompt": prompt + face, "cvec": cv,
                        "choices": [sep + c + "<turn|>" for c in choices]})["choice_logprobs"]
        return {"face": face, "logprobs": {c: lp[sep + c + "<turn|>"] for c in choices}}

    def read_feel(self, text):
        """life_sim.py read_feel: the text as an answer to "Write a note.", no drip, rooms 24-36"""
        prompt = self.render([{"role": "user", "content": "Write a note."}])
        H = self._states(prompt + text, prompt, READ, "mean")
        out = []
        for k in ("valence", "arousal"):
            x = [(H[i] @ self.dial[k][L] - self.spread[k + "_rest"][L]) / self.spread[k][L] for i, L in enumerate(READ)]
            out.append(float(sum(x) / len(x)))
        return out

    def close(self):
        self.p.stdin.close(); self.p.wait()


def dials(dpath, vkey, akey, out):
    import torch
    D = torch.load(dpath, map_location="cpu", weights_only=False)
    u = {k: (D[key].float() / D[key].float().norm(dim=-1, keepdim=True)).numpy() for k, key in
         (("valence", vkey), ("arousal", akey))}
    np.savez(out, **u)
    print("saved", out, {k: v.shape for k, v in u.items()})


def check(gguf, dials_npz, life_json, scenario, way, out):
    L = json.load(open(life_json))
    lives = L[scenario][way]
    b = Brain(gguf, dials_npz)
    b.calibrate()
    rows = []
    for li, life in enumerate(lives if isinstance(lives, list) else [lives]):
        for t in life["ticks"]:
            note = t.get("thought") or t.get("note")
            cloud = t.get("thought_read") or t.get("thought_reads")
            if not note or cloud is None:
                continue
            rows.append({"life": li, "when": t["when"], "cloud": cloud, "mac": [round(x, 3) for x in b.read_feel(note)],
                         "note": note[:120]})
    b.close()
    c = np.array([r["cloud"][:2] for r in rows], float); m = np.array([r["mac"] for r in rows], float)
    summary = {k: {"corr": round(float(np.corrcoef(c[:, i], m[:, i])[0, 1]), 3),
                   "mean_abs_diff": round(float(np.abs(c[:, i] - m[:, i]).mean()), 3),
                   "cloud_range": [round(float(c[:, i].min()), 2), round(float(c[:, i].max()), 2)]}
               for i, k in enumerate(("valence", "arousal"))}
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    {"dials": dials, "check": check}[sys.argv[1]](*sys.argv[2:])
