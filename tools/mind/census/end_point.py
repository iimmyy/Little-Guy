#!/usr/bin/env python3
"""Where does he decide to stop thinking? (2026-11-03: at paragraph ends the chance of ending the thinking is ~0, in
wrap-ups and loops alike, so the stop is decided somewhere else.) For his finished runs on the train split: put his
thinking back with three possible joins before the end tag ("", "\n", "\n\n") and read the chance of "</think>" at
the join, to learn which token carries the stop. Then, at every sentence end after his first right statement (in
finished runs: the last one is where he did stop), the chance of "</think>" at that point under the found join, for
finished runs (last sentence vs the earlier ones) and for cut-off runs (all sentence ends after the answer).
usage: end_point.py <model folder> <out.txt> <runs.jsonl> [...] [n=40 tail=0 dev=cuda]
"""
import json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3
from situation_probe import TAIL
from natural_error import whole

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
mdir, out, runs = args[0], args[1], args[2:]
N, DEV = int(opt.get("n", 40)), opt.get("dev", "cuda")
if opt.get("tail") == "0":                              # the bare question (Immy: no prompts)
    TAIL = ""
p = _lfm3(DEV, mdir)
tok, m = p.tok, p.model
END = tok("</think>", add_special_tokens=False).input_ids
assert len(END) == 1, END
END = END[0]


def first_statement(think, A):
    mm = re.search(rf"(?:=|\bis\b|\bare\b|\bget\b|\bgives\b|\bequals\b)\s*\$?\\?\(?\s*{A}(?![\d.,/])", think)
    return mm.end() if mm else None


def head_of(t):
    return tok.apply_chat_template([{"role": "user", "content": t["q"] + TAIL}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) + "\n"


def p_end(ids):
    with torch.no_grad():
        hs = m.model(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0]
    return hs


items = []
for f in runs:
    for t in map(json.loads, open(f)):
        A = whole(str(t["answer"]))
        if A is None or A < 2 or not t["thinking"]:
            continue
        st = first_statement(t["thinking"], A)
        if st is not None:
            items.append((t, st))
fin = [x for x in items if x[0]["finish"] == "stop"][:N]
cut = [x for x in items if x[0]["finish"] != "stop"][:N]
L = [f"stop point on {os.path.basename(mdir.rstrip('/'))}: {len(fin)} finished and {len(cut)} cut-off runs with the right answer stated"]
joins = {}
for j in ("", "\n", "\n\n"):
    pr = []
    for t, st in fin:
        ids = tok(head_of(t) + t["thinking"] + j, add_special_tokens=False).input_ids
        hs = p_end(ids)
        pr.append(float(torch.softmax(m.lm_head(hs[-1]).float(), -1)[END]))
    joins[j] = float(np.median(pr))
L.append("  chance of </think> right after the end of his thinking, by join: " + ", ".join(f"{j!r} {v:.3f}" for j, v in joins.items()))
J = max(joins, key=joins.get)
SENT = re.compile(r"[.!?)\]]\s*(?=\n|$)")                    # sentence ends at a line end (where a paragraph could close)


def sentence_ends(t, st):
    th = t["thinking"]
    return [mm.end() for mm in SENT.finditer(th) if mm.end() > st]


rows = {"finished: the sentence where he stopped": [], "finished: earlier sentence ends after the answer": [],
        "cut off: sentence ends after the answer": []}
for kind, group in (("fin", fin), ("cut", cut)):
    for t, st in group:
        th = t["thinking"]
        ends = sentence_ends(t, st)
        if not ends:
            continue
        head = head_of(t)
        enc = tok(head + th + (J if kind == "fin" else ""), add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        h0 = len(head)
        hs = p_end(ids)
        pos = []
        for e in ends:
            k = max(i for i, (a, b) in enumerate(offs) if b <= h0 + e) if any(b <= h0 + e for a, b in offs) else None
            if k is not None:
                pos.append(k)
        if kind == "fin":
            pos.append(len(ids) - 1)                          # the join itself, where he did stop
        pos = sorted(set(pos))
        pe = torch.softmax(m.lm_head(hs[pos]).float(), -1)[:, END].cpu().numpy()
        if kind == "fin":
            rows["finished: the sentence where he stopped"].append(float(pe[-1]))
            rows["finished: earlier sentence ends after the answer"] += [float(x) for x in pe[:-1]]
        else:
            rows["cut off: sentence ends after the answer"] += [float(x) for x in pe]
for k, v in rows.items():
    v = np.array(v)
    if len(v):
        L.append(f"  {k}: n {len(v)}, chance of </think> median {np.median(v):.3f}, mean {v.mean():.3f}, share over 0.1 {np.mean(v > 0.1):.0%}")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
