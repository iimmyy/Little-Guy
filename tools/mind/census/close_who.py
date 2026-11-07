#!/usr/bin/env python3
"""What closes him? Reads a touched build at the last word of each run's thinking and at the last paragraph end before it, with its added neurons silenced one group at a time.
Silencing a neuron means setting its output column to zero, so what is left at that word is everything else in the model.
usage: close_who.py <build HF folder> <plain HF folder> <out.jsonl> <jobs.jsonl> groups=name:layer:i,i;name:layer:i,..."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
build, plain, out, jobs = args
DEV = opt.get("dev", "cuda")
G = {}
for g in opt["groups"].split(";"):
    n, l, ii = g.split(":")
    G[n] = (int(l), [int(i) for i in ii.split(",")])
p = _lfm3(DEV, build)
tok, m = p.tok, p.model
END = tok("</think>", add_special_tokens=False).input_ids[0]
THUS = tok(" Thus", add_special_tokens=False).input_ids[0]
THUS2 = tok("Thus", add_special_tokens=False).input_ids[0]
orig = {(l, i): m.model.layers[l].feed_forward.w2.weight[:, i].clone() for l, ii in G.values() for i in ii}
for (l, i), v in orig.items():
    print("neuron", l, i, "output norm", round(float(v.float().norm()), 4), flush=True)


def silence(names):
    with torch.no_grad():
        for (l, i), v in orig.items():
            m.model.layers[l].feed_forward.w2.weight[:, i] = v
        for n in names:
            l, ii = G[n]
            for i in ii:
                m.model.layers[l].feed_forward.w2.weight[:, i] = 0


def read(ids, at):
    """chance of the close word and of Thus at each position in `at`"""
    with torch.no_grad():
        hs = m.model(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0]
        o = []
        for a in at:
            pr = torch.softmax(m.lm_head(hs[a]).float(), -1)
            o.append({"close": float(pr[END]), "thus": float(pr[THUS] + pr[THUS2])})
    return o


VARS = [("full", [])] + [(f"no_{n}", [n]) for n in G] + [("no_all", list(G))]
done = set()
if os.path.exists(out):
    done = {(json.loads(l)["run"], json.loads(l)["id"]) for l in open(out)}
with open(out, "a") as f:
    for c, j in enumerate(map(json.loads, open(jobs))):
        if (j["run"], j["id"]) in done:
            continue
        head = tok.apply_chat_template([{"role": "user", "content": j["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        th = j["thinking"].rstrip()
        enc = tok(head + "\n" + th + opt.get("tail", "").encode().decode("unicode_escape"), add_special_tokens=False, return_offsets_mapping=True)    # the server strips the paragraph break he wrote before closing
        ids = enc.input_ids
        cut = (head + "\n" + th).rfind("\n\n")
        par = next((i for i, (a, b) in enumerate(enc.offset_mapping) if a >= cut), len(ids) - 1) - 1
        if cut <= len(head) or par < 1:
            par = len(ids) - 1
        last3 = tok.decode(ids[-3:])
        row = dict(id=j["id"], run=j["run"], f5_grade=j.get("f5_grade"), base_grade=j.get("base_grade"), n=len(ids), words=len(th.split()), last=th[-60:], par_from_end=len(ids) - 1 - par)
        for vn, gs in VARS:
            silence(gs)
            r = read(ids, [len(ids) - 1, par])
            row[vn] = r
        silence([])
        f.write(json.dumps(row) + "\n"); f.flush()
        if c % 25 == 0:
            print(c, last3[-30:].replace("\n", " "), row["full"][0], flush=True)
print("WHO_DONE", flush=True)
