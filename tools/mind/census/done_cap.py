#!/usr/bin/env python3
"""Step 1 of the hand-in cell (Immy 2026-11-03: much more gain). Reading only, on the bare question, build split only (problems in the
eval set passed as skip= stay out). His runs are read at the two points where a commit sentence starts: paragraph ends (next word
could be "Thus") and the opener word itself (next word could be "final"), plus random ordinary words so a gate can stay silent.
Saved: the stream at each layer's MLP input, words so far, words since he first stated the right answer (any form; -1 if not yet),
the chance of each commit word next. kind: 0 paragraph end, 1 opener word, 2 ordinary word, 3 any word in the reply. Answer matching covers every form, because half his words are on non-whole answers.
usage: done_cap.py <model> <out.npz> <runs.jsonl> [...] [skip=eval.jsonl layers=14,17,20,23 per=25 ord=8 dev=cuda]
"""
import json, os, random, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patients import _lfm3

args = [a for a in sys.argv[1:] if "=" not in a]
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
mdir, outz, runs = args[0], args[1], args[2:]
LAY = [int(x) for x in opt.get("layers", "14,17,20,23").split(",")]
PER, ORD, DEV = int(opt.get("per", 25)), int(opt.get("ord", 8)), opt.get("dev", "cuda")
skip = {json.loads(l)["id"] for l in open(opt["skip"])} if "skip" in opt else set()
OPENERS = ("Thus", "So", "Therefore", "Hence")


def norm(s):
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("$", ""), (" ", ""), ("{,}", "")):
        s = s.replace(a, b)
    return s


def spot_char(th, ans):
    """char offset right after the first place the right answer is stated, in any form; None if never"""
    a = norm(str(ans))
    if not a or (len(a) < 2 and not a.isdigit()):
        return None
    forms = {a}
    m_ = re.fullmatch(r"(-?)\\frac\{(\d+)\}\{(\d+)\}", a)
    if m_:
        forms.add(f"{m_.group(1)}{m_.group(2)}/{m_.group(3)}")
    keep = [i for i, ch in enumerate(th) if norm(ch)]
    S = "".join(th[i] for i in keep)
    best = None
    for f in forms:
        for mm in re.finditer(re.escape(f), S):
            nxt, prv = S[mm.end():mm.end() + 1], S[mm.start() - 1:mm.start()] if mm.start() else ""
            if (f[-1].isdigit() and nxt.isdigit()) or (f[0].isdigit() and (prv.isdigit() or prv == ".")) or nxt in ("^", "+", "x", "y") or (nxt == "/" and "/" not in f):
                continue
            ctx = S[max(0, mm.start() - 6):mm.start()]
            if any(c in ctx for c in ("=", "is", "get", "are", "gives", "equals", "≈", "boxed{", "answer")):
                e = keep[mm.end() - 1] + 1
                best = e if best is None else min(best, e)
                break
    return best


p_ = _lfm3(DEV, mdir)
tok, m = p_.tok, p_.model
got = {}
for L in LAY:
    m.model.layers[L].feed_forward.register_forward_pre_hook(lambda mod, a, L=L: got.__setitem__(L, a[0][0].float()))
ids_of = lambda w: tok(w, add_special_tokens=False).input_ids[0]
THUS, FIN, ANS, END = ids_of("Thus"), ids_of(" final"), ids_of(" answer"), ids_of("</think>")
OPEN_IDS = [ids_of(w) for w in OPENERS]
rng = random.Random(0)
X = {L: [] for L in LAY}
KIND, SINCE, POS, FIN_, G, GID, PTH, POP, PFI, PAN, RUN, PE = [], [], [], [], [], [], [], [], [], [], [], []
nrun = 0
for f in runs:
    for t in map(json.loads, open(f)):
        th = t["thinking"]
        if t["id"] in skip or not th or len(th.split()) < 20:
            continue
        head = tok.apply_chat_template([{"role": "user", "content": t["q"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
        # the reply counts too: a cell must be silent there (kind 3), or a clock that has passed its bound keeps writing "</think>" into the reply
        rep = "\n</think>\n\n" + t["reply"] if t["finish"] == "stop" and t.get("reply") else ""
        full = head + th + rep
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, off = enc.input_ids, enc.offset_mapping
        h0, h1 = len(head), len(head) + len(th)
        sp = spot_char(th, t["answer"])
        par, opn, ord_, rp, rq = [], [], [], [], []
        for i in range(len(ids) - 1):
            a, b = off[i]
            if a < h0:
                continue
            txt = full[a:b]
            if a >= h1:
                (rp if "\n\n" in txt or txt.strip() in OPENERS else rq).append(i)
            elif "\n\n" in txt:
                par.append(i)
            elif txt.strip() in OPENERS and i > 0 and "\n" in full[off[i - 1][0]:off[i - 1][1]]:
                opn.append(i)
            else:
                ord_.append(i)
        rng.shuffle(par); rng.shuffle(opn); rng.shuffle(ord_); rng.shuffle(rp); rng.shuffle(rq)
        chosen = [(i, 0) for i in par[:PER]] + [(i, 1) for i in opn[:PER]] + [(i, 2) for i in ord_[:ORD]] + [(i, 3) for i in rp[:PER]] + [(i, 3) for i in rq[:ORD]]
        if not chosen:
            continue
        with torch.no_grad():
            hs = m.model(input_ids=torch.tensor([ids], device=DEV)).last_hidden_state[0]
            pos = torch.tensor([c[0] for c in chosen], device=DEV)
            pr = torch.softmax(m.lm_head(hs[pos]).float(), -1).cpu().numpy()
        for L in LAY:
            X[L].append(got[L][pos].cpu().numpy().astype(np.float32))
        for (i, k), row in zip(chosen, pr):
            e = off[i][1] - h0
            so_far = len(full[h0:h0 + e].split())
            since = -1 if sp is None or e < sp else len(full[h0 + sp:h0 + e].split())
            KIND.append(k); SINCE.append(since); POS.append(so_far); FIN_.append(t["finish"] == "stop")
            G.append(t["id"]); RUN.append(nrun)
            PTH.append(row[THUS]); POP.append(row[OPEN_IDS].sum()); PFI.append(row[FIN]); PAN.append(row[ANS]); PE.append(row[END])
        nrun += 1
        if nrun % 40 == 0:
            print(f"{nrun} runs, {len(KIND)} points", flush=True)
np.savez(outz, **{f"x{L}": np.concatenate(X[L]) for L in LAY}, kind=np.array(KIND), since=np.array(SINCE), pos=np.array(POS), fin=np.array(FIN_), gid=np.array(G),
         run=np.array(RUN), p_thus=np.array(PTH), p_open=np.array(POP), p_final=np.array(PFI), p_answer=np.array(PAN), p_end=np.array(PE))
print("CAP_DONE", nrun, len(KIND))
