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
  read_feel(text)      how pleasant / keyed up a text reads inside him, in natural steps (rooms 24-36)
"""
import json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LG_MIND = os.environ.get("LG_MIND", os.path.join(HERE, "..", "..", "engine", "lg-mind", "build", "llama-lg-mind"))
DRIP = range(18, 41)                       # rooms the feelings drip into
READ = range(24, 37)                       # rooms his notes are read from
CAL = ["Hey.", "How's it going?", "Tell me about octopuses.", "Plan me a quick dinner with eggs and rice."]


def chat_template(gguf_path):
    """the brain's own chat template (from its GGUF), rendered the way transformers does"""
    sys.path.insert(0, os.path.join(HERE, "..", "..", ".run", "lcpp-src", "gguf-py"))
    from gguf import GGUFReader
    import jinja2
    f = GGUFReader(gguf_path).fields["tokenizer.chat_template"]
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    def fail(m):
        raise ValueError(m)
    env.globals["raise_exception"] = fail
    t = env.from_string(bytes(f.parts[f.data[0]]).decode())
    return lambda msgs: t.render(messages=msgs, add_generation_prompt=True, bos_token="<bos>")


class Brain:
    def __init__(self, gguf, dials_npz, ctx=4096, log=None):
        d = np.load(dials_npz)
        self.dial = {"valence": d["valence"].astype(np.float32), "arousal": d["arousal"].astype(np.float32)}
        self.render = chat_template(gguf)
        self.tmp = tempfile.mkdtemp(prefix="lg-brain-")
        self.p = subprocess.Popen([LG_MIND, "-m", gguf, "-ngl", "99", "-c", str(ctx)], stdin=subprocess.PIPE,
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
