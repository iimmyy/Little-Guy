#!/usr/bin/env python3
"""How does a build's clock neuron read length, paragraph by paragraph? For runs from any set: at each paragraph break, its gate, its output, and the chance of the close word, with the true word count beside them.
usage: clock_profile.py <build HF> <out.jsonl> name=<layer>:<neuron> <label>=<runs.jsonl>[,<n>] ... [maxwords=2600 cap=<tokens>]"""
import json, os, random, re, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
build, out = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
LB, NB = (int(x) for x in opt.pop("name").split(":"))
MAXW = int(opt.pop("maxwords", 2600))
DEV = opt.pop("dev", "cuda")
p = _lfm3(DEV, build); tok, m = p.tok, p.model
END = tok("</think>", add_special_tokens=False).input_ids[0]
ff = m.model.layers[LB].feed_forward
got = {}
ff.register_forward_pre_hook(lambda mod, a: got.__setitem__("x", a[0][0]))
rng = random.Random(4)
with open(out, "w") as w:
    for label, spec in opt.items():
        path, _, n = spec.partition(",")
        J = [json.loads(l) for l in open(path)]
        J = [j for j in J if j.get("thinking") and len(j["thinking"].split()) >= 60]
        rng.shuffle(J)
        J = J[: int(n)] if n else J
        for c, j in enumerate(J):
            th = j["thinking"].rstrip()
            ws = th.split()
            if len(ws) > MAXW:
                k = th.find(ws[MAXW]); th = th[:k]
            head = tok.apply_chat_template([{"role": "user", "content": j["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
            text = head + th + "\n\n"
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ends = np.array([b for a, b in enc.offset_mapping])
            marks = [mm.end() for mm in re.finditer(r"\n\n", text) if mm.end() > len(head)]
            idx = sorted({int(np.searchsorted(ends, e, side="left")) for e in marks if e <= ends[-1]})
            with torch.inference_mode():
                hs = m.model(input_ids=torch.tensor([enc.input_ids], device=DEV)).last_hidden_state[0]
            x = got["x"][idx].float()
            g = x @ ff.w1.weight[NB].float(); l = x @ ff.w3.weight[NB].float()
            a = F.silu(g) * l
            pc = torch.softmax(m.lm_head(hs[idx]).float(), -1)[:, END]
            for i, gg, ll, aa, pp in zip(idx, g.tolist(), l.tolist(), a.tolist(), pc.tolist()):
                wc = len(tok.decode(enc.input_ids[:i + 1]).split()) - len(head.split())
                w.write(json.dumps(dict(set=label, id=j["id"], run=j.get("run"), tok=i, words=wc, gate=gg, level=ll, act=aa, close=pp, finish=j.get("finish"))) + "\n")
            if c % 25 == 0:
                print(label, c, len(idx), flush=True)
print("PROFILE_DONE", flush=True)
