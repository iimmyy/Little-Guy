#!/usr/bin/env python3
"""Belief PAM (Immy 2026-10-31: a principle bolts on like a PAM; build it on the biggest unused ligand). The "I have the
right answer" signal is written on his page as clearly as anything (ligand_use.py: AUROC 0.97) but his wrap-up drive
barely follows it (+0.03), and the doubt crowd in layers 18-21 votes "But" at every fork (fork_votes.py).

The belief gate (bake_belief.py) added a new neuron. This adds nothing: it changes the binding site of the doubt
receptors that already exist. Each chosen receptor's gate read (w1 row) gets a component along the belief line at its
own input, so its activation curve shifts down by an amount set by belief:
    gate_j(x) -> gate_j(x) - beta_j * (belief(x) - theta)
The squash (silu) is near flat where the receptor rests, so the shift only matters where the receptor is already
engaged by its own trigger (a fork where "But" is on the table): the modulator does nothing without the agonist, the
PAM property. The offset rides on the stream's near-constant line (the "spike" line, the model's own bias carrier).

No training, no new neurons, no run-time code: the same file shape with a few w1 rows changed.

usage: pam_belief.py <hf folder> <answers jsonl[,more]> <out folder> [layers=18-21 k=24 shift=3 theta=0.5 neutral=0|1 n=110 dev=cuda]
neutral=1 fits the belief line to read theta (undecided) at ordinary thinking and reply words too, so the shift stays ~0 off the forks.
anchor=starts (with neutral=1) also anchors sentence starts before any answer is stated to theta: "no answer yet" is
undecided, only a stated wrong answer should read low (p3 added doubt at ordinary mid-reasoning conclusions).
drop=20:7254,20:5676 leaves those layer:neuron receptors out of the top k (the ones turning digits in the weigh-in).
claims=phrase (default, LFM3's "answer is X") or value (any model: wherever he writes the right answer or his final
wrong one; see claims()).
model=lfm3 (default) or e4b: the same recipe on another model (the universality check). <hf folder> is that model's
folder; for e4b the changed rows are written as a patch (pam_rows.safetensors) plus a full patched copy of the folder.
"""
import json, os, re, shutil, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../expert"))
ARGS, sys.argv[1:] = sys.argv[1:], []                     # grade.py grades its own arguments on import
from grade import right
from patients import _lfm3
from probe_math import ridge

# where one model keeps the parts the recipe touches (a new model is one entry here)
MODEL = next((a.split("=", 1)[1] for a in ARGS[3:] if a.startswith("model=")), "lfm3")
ARCH = {"lfm3": dict(mlp="feed_forward", gate="w1", up="w3", down="w2", post="post_ffn_norm", pre="ffn_norm",
                     opnorm="operator_norm", act=F.silu, think_open="\n", think_close="\n</think>\n\n"),
        "e4b": dict(mlp="mlp", gate="gate_proj", up="up_proj", down="down_proj", post="post_feedforward_layernorm",
                    pre="pre_feedforward_layernorm", opnorm="input_layernorm",
                    act=lambda g: F.gelu(g, approximate="tanh"), think_open="<|channel>thought\n", think_close="\n<channel|>"),
        # Qwen3: no norm after the MLP (the write goes straight into the stream); its template opens <think> itself
        "qwen3": dict(mlp="mlp", gate="gate_proj", up="up_proj", down="down_proj", post=None, pre="post_attention_layernorm",
                      opnorm="input_layernorm", act=F.silu, think_open="", think_close="\n</think>\n\n")}[MODEL]

src, ans, out = ARGS[:3]
opt = dict(a.split("=", 1) for a in ARGS[3:])
lo, hi = (int(x) for x in opt.get("layers", "18-21").split("-"))
RL = list(range(lo, hi + 1))
K, SHIFT, TH = int(opt.get("k", 24)), float(opt.get("shift", 3)), float(opt.get("theta", 0.5))
N, DEV = int(opt.get("n", 110)), opt.get("dev", "cuda")
TAIL = "" if opt.get("tail") == "0" else "\n\nEnd with a line: Answer: <your answer>"     # tail=0: the bare question (no-prompts rule), with bare runs as answers
CLAIM = re.compile(r"\\boxed\{([^{}]{1,40})\}|(?:answer is|answer =|answer:)\s*\$?([^\n$]{1,40}?)\$?(?=\.(?:\s|$)|\n|$)", re.I)
# (2026-11-01: the old end "(?=[.\n]|$)" stopped at a decimal point, "answer is 1.25" read as "1"; p1-p4 used it)
DOUBT = re.compile(r"(But|Wait|However|Hmm|Let me|Let's|Actually|Alternatively|Double|Check|Hold on)\b", re.I)
HANDOFF = re.compile(r"(I'll (produce|write|output|give|present|answer)|We need to (end|output|produce|write|present)"
                     r"|Now (we need to )?(output|produce|write)|So (the )?final (output|line)|Thus,? final\.?\s*$"
                     r"|Thus,? I'll|End with|So final answer|Thus final answer)", re.I)
STARTS = opt.get("anchor", "") == "starts"
HAND_IDS, DOUBT_IDS = [49, 1672, 2083, 8175, 21009], [1938, 2375, 3911, 7731, 8952]    # the belief gate's openers

if MODEL == "e4b":
    os.environ["E4B_MASTER"] = src.rstrip("/")
    from e4b_mac import load as e4b_load
    m, tok = e4b_load(DEV)
    FINAL = m.model.norm
elif MODEL == "qwen3":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).to(DEV).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(src)
    FINAL = m.model.norm
else:
    p = _lfm3(DEV, src)
    tok, m = p.tok, p.model
    FINAL = m.model.embedding_norm
lay = m.model.layers
MLP = lambda L: getattr(lay[L], ARCH["mlp"])
WT = lambda L, nm: getattr(MLP(L), ARCH[nm]).weight
POST = lambda L: getattr(lay[L], ARCH["post"]) if ARCH["post"] else None   # None: the MLP writes the stream directly
POSTW = lambda L: POST(L).weight.float() if POST(L) is not None else torch.ones(m.config.hidden_size, device=DEV)
# Gemma 4 multiplies the whole stream by a per-layer scalar after each layer: a write at layer L reaches the end scaled
# by the product of the scalars from L on (LFM3 has none)
SC = {L: float(np.prod([float(lay[i].layer_scalar) for i in range(L, len(lay))])) if hasattr(lay[L], "layer_scalar") else 1.0
      for L in RL}
if MODEL != "lfm3":                                       # the same opener words, in this model's tokenizer
    def one_token(words):
        ids = []
        for w in words:
            for v in (w, " " + w):
                t = tok(v, add_special_tokens=False).input_ids
                if len(t) == 1 and t[0] not in ids:
                    ids.append(t[0])
        return ids
    HAND_IDS, DOUBT_IDS = one_token(["I", "So", "We", "Now", "Thus"]), one_token(["But", "However", "Let"])
print(f"model {MODEL}: hand-in openers {[tok.decode([i]) for i in HAND_IDS]}, doubt openers {[tok.decode([i]) for i in DOUBT_IDS]}",
      flush=True)
rows = [r for path in ans.split(",") for r in map(json.loads, open(path)) if r["thinking"]][:N]


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


VALUE = opt.get("claims", "phrase") == "value"
SNIP = re.compile(r"\$\$(.{1,80}?)\$\$|\$([^$\n]{1,80})\$|\\\((.{1,80}?)\\\)|\\\[(.{1,120}?)\\\]|\\boxed\{([^{}]{1,40})\}"
                  r"|(?:=|\bis|:)\s*([-\w\\{}^/√.+()]{1,30}?)\s*(?=[.,;]?(?:\s|$))", re.S)
_CL = {}


def claims(think, r):
    """Answer claims in his thinking: (end of the claim, right?, which answer). phrase: "answer is X" / boxed (LFM3's
    habit). value (any model): every spot where he writes the right answer, or the wrong answer he ends up handing in
    (answer key and his own final answer only, no phrasing; one-character answers are skipped, every "= 2" would count),
    plus his "answer is X" phrases where he has that habit, graded with unicode read as TeX (the phrase grader called
    "-I", "(-∞, 2) ∪ (3, ∞)" and "-√3" wrong). LFM3 almost never hands in a wrong answer (1 of ~490 finished runs), so
    its wrong claims are the mid-run ones ("answer is X. But wait")."""
    k = id(r)
    if k in _CL:
        return _CL[k]
    out = []
    if not VALUE:
        for mt in CLAIM.finditer(think):
            x = next(g for g in mt.groups() if g is not None).strip().rstrip(".").strip()
            if x:
                out.append((mt.end(), bool(right({**r, "reply": x}, x)), re.sub(r"[\s$\\{}]|boxed", "", x)))
    else:
        from math_verify import parse, verify
        from grade import right_reply
        gold = parse("$" + r["answer"] + "$")
        refs = [("right", gold)] if len(r["answer"].strip()) > 1 and gold else []
        m_ = list(re.finditer(r"(?i)answer\W*[:：]", r["reply"] or ""))
        fin = (r["reply"][m_[-1].end():].strip().strip("*").splitlines() or [""])[0].strip().rstrip(".") if m_ else ""
        if fin and len(fin) > 1 and r["finish"] == "stop" and not right_reply(r, r["reply"]):
            g = parse("$" + fin.strip("$") + "$") or parse(fin)
            if g:
                refs.append(("wrong", g))
        for mt in SNIP.finditer(think) if refs else []:
            x = next(g for g in mt.groups() if g is not None).strip()
            cands = {x, x.split("=")[-1].strip()} if "=" in x else {x}
            for name, ref in refs:
                hit = False
                for c in cands:
                    try:
                        pc = parse("$" + c + "$")
                        hit = bool(pc) and bool(verify(ref, pc))
                    except Exception:
                        hit = False
                    if hit:
                        break
                if hit:
                    out.append((mt.end(), name == "right", name))
                    break
        if gold:                                           # his own "answer is X" habit, graded with unicode as TeX
            for mt in CLAIM.finditer(think):
                x = next(g for g in mt.groups() if g is not None).strip().rstrip(".").strip()
                if (not x or "<your answer>" in x or "..." in x or "…" in x or x.endswith("{") or any(abs(mt.end() - e) < 3 for e, _, _ in out)
                        or re.search(r"(?<!\\)\b[a-zA-Z]{4,}", x)):       # an answer in words ("all real numbers > 2")
                    continue
                tex = re.sub(r"√\s*\(?(\w+)\)?", r"\\sqrt{\1}", x)
                for a_, b_ in (("π", "\\pi"), ("∞", "\\infty"), ("∪", "\\cup"), ("≤", "\\le"), ("≥", "\\ge"), ("·", "\\cdot"), ("×", "\\times")):
                    tex = tex.replace(a_, b_)
                try:
                    px = parse("$" + tex + "$")
                    ok = bool(px) and bool(verify(gold, px))
                except Exception:
                    continue
                if px:
                    out.append((mt.end(), ok, "right" if ok else re.sub(r"[\s$\\{}]|boxed", "", x)))
        out.sort()
    _CL[k] = out
    return out


# ---- 0. the words: every claim's fork (the word before the next sentence), other thinking words, the reply
SEQ, lab = [], []                      # lab: (row, role 0 thinking / 1 fork / 2 reply, right, prior statements, kind)
for n, r in enumerate(rows):
    pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
    think = ARCH["think_open"] + r["thinking"]
    enc = tok(think, add_special_tokens=False, return_offsets_mapping=True)
    th, starts = enc.input_ids[:6000], np.array([s for s, _ in enc.offset_mapping[:6000]])
    pos, seen = {}, {}
    for e, ok, key in claims(think, r):
        prior = seen.get(key, 0); seen[key] = prior + 1
        while e < len(think) and not (think[e] == "\n" or think[e] == "." and (e + 1 == len(think) or think[e + 1] in " \n")):
            e += 1
        s = e + 1
        while s < len(think) and think[s] in " \n":
            s += 1
        if s >= len(think) - 2:
            continue
        t = int(np.searchsorted(starts, s, side="right")) - 1
        if not 1 <= t < len(th):
            continue
        nxt = think[s:s + 60]
        pos[t - 1] = (n, 1, ok, prior, "hand" if HANDOFF.match(nxt) else "doubt" if DOUBT.match(nxt) else "other")
    T = len(th)
    if STARTS:                                             # sentence starts before any answer is stated: "not yet", not "wrong"
        c0 = next((e for e, _, _ in claims(think, r)), len(think))
        ss = [int(np.searchsorted(starts, mm.end(), side="right")) - 1 for mm in re.finditer(r"(?:\n|[.!?] )\s*(?=[A-Z])", think)
              if mm.end() < c0]
        ss = sorted({t for t in ss if 2 <= t < T})
        for t in [ss[i] for i in np.linspace(0, len(ss) - 1, min(24, len(ss))).round().astype(int)] if ss else []:
            if t - 1 not in pos and t not in pos:
                pos[t - 1] = (n, 3, False, 0, "")
    for t in sorted(set(int(v) for v in np.exp(np.random.RandomState(T).uniform(0, np.log(max(2, T - 1)), 32)))):
        if t < T and t not in pos and t + 1 not in pos:
            pos[t] = (n, 0, False, 0, "")
    rp = tok(ARCH["think_close"] + r["reply"], add_special_tokens=False).input_ids[:240] if r["finish"] == "stop" else []
    for k_ in range(0, len(rp), 4):
        pos[T + k_] = (n, 2, False, 0, "")
    ks = sorted(pos)
    SEQ.append((pr + th + rp, torch.tensor([len(pr) + k_ for k_ in ks])))
    lab += [pos[k_] for k_ in ks]
R_ = np.array([l[0] for l in lab]); ROLE = np.array([l[1] for l in lab]); OK = np.array([l[2] for l in lab])
PRIOR = np.array([l[3] for l in lab])
fork = ROLE == 1
groups = {"right fork, stated before": fork & OK & (PRIOR >= 1), "right fork, first statement": fork & OK & (PRIOR == 0),
          "wrong fork": fork & ~OK, "other thinking": ROLE == 0, "reply": ROLE == 2}
if STARTS:
    groups["sentence start, no answer yet"] = ROLE == 3
print(f"{len(rows)} answers: " + ", ".join(f"{k} {int(v.sum())}" for k, v in groups.items()), flush=True)

# ---- 1. read the receptor layers: their input (after the auto-level), every neuron's gate and firing, output size
gf = FINAL.weight.float()
dvec = gf * (m.lm_head.weight[HAND_IDS].float().mean(0) - m.lm_head.weight[DOUBT_IDS].float().mean(0))   # hand-in minus doubt
cap_all = {L: {"x": [], "g": [], "vote": []} for L in RL}
GAP = []                                   # hand-in minus doubt score at each word (the whole model's verdict)
# every reader of the per-neuron gate / vote looks only at fork words, so only those rows are kept (all words x every
# neuron x 12 layers OOM-killed the 30 GB box on E4B); FK() maps a mask over all words to these rows
FORK_T = torch.tensor(fork)


def FK(mask):
    mask = torch.as_tensor(mask)
    assert not bool((mask & ~FORK_T).any()), "per-neuron captures exist only at fork words"
    return mask[FORK_T]


off_ = 0
with torch.no_grad():
    for ids, pos in SEQ:
        frk = FORK_T[off_:off_ + len(pos)]; off_ += len(pos)
        cap = {}
        hs = []
        for L in RL:
            hs.append(MLP(L).register_forward_pre_hook(lambda mod, a, L=L: cap.__setitem__(("x", L), a[0][0, pos].float())))
            if POST(L) is not None:
                hs.append(POST(L).register_forward_pre_hook(lambda mod, a, L=L: cap.__setitem__(("y", L), a[0][0, pos].float())))
        hs.append(FINAL.register_forward_pre_hook(lambda mod, a: cap.__setitem__("fin", a[0][0, pos].float())))
        m.model(input_ids=torch.tensor([ids], device=DEV))
        for h in hs:
            h.remove()
        rf = cap["fin"].pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
        GAP.append(((cap["fin"] / rf) @ dvec).cpu())
        for L in RL:
            x = cap[("x", L)]
            g = x @ WT(L, "gate").float().T
            act = ARCH["act"](g) * (x @ WT(L, "up").float().T)
            ry = cap[("y", L)].pow(2).mean(-1, keepdim=True).add(1e-5).sqrt() if POST(L) is not None else 1.0
            wd = SC[L] * (POSTW(L)[:, None] * WT(L, "down").float() * dvec[:, None]).sum(0)   # per neuron
            cap_all[L]["x"].append(x.half().cpu()); cap_all[L]["g"].append(g[frk.to(g.device)].half().cpu())
            cap_all[L]["vote"].append((act * wd[None] / ry / rf)[frk.to(g.device)].half().cpu())   # each neuron's push toward hand-in (score points)
for L in RL:
    for k_ in cap_all[L]:                      # per-neuron captures stay half; slices go float where used
        cat_ = torch.cat(cap_all[L][k_])
        cap_all[L][k_] = cat_.float() if k_ == "x" else cat_
GAP = torch.cat(GAP).float()

# ---- 2. the belief line at each receptor layer's input (forks: right vs wrong), held out by answer, then refit
lines = [f"belief PAM on {os.path.basename(src.rstrip('/'))}: receptor layers {RL}, k {K}, shift {SHIFT}, theta {TH}"]
ev_, od_ = R_ % 2 == 0, R_ % 2 == 1
NEUTRAL = opt.get("neutral", "0") == "1"
rs_ = np.random.RandomState(0)
NEU = np.zeros(len(ROLE), bool)                            # ordinary words for the neutral anchor: as many as forks, both kinds
for role in (0, 2, 3):
    ix = np.flatnonzero(ROLE == role)
    if len(ix):
        NEU[rs_.choice(ix, min(len(ix), int(fork.sum()) // 2), replace=False)] = True
lines[0] += f", neutral anchor {'on' if NEUTRAL else 'off'}" + (" (+ sentence starts before any answer)" if STARTS else "")
BL = {}
for L in RL:
    Xn = cap_all[L]["x"].numpy()
    if NEUTRAL:                                            # also read "undecided" (theta) at ordinary words, so the shift stays ~0 there
        tr = fork | NEU
        y_ = np.where(fork, OK.astype(float), TH)
    else:
        tr, y_ = fork, OK.astype(float)
    pb, _, _, _ = ridge(Xn[ev_ & tr], y_[ev_ & tr], Xn[od_ & fork])
    _, _, _, (wb, bb) = ridge(Xn[tr], y_[tr], Xn[fork][:1])
    X = cap_all[L]["x"]
    mu, sd = X.mean(0), X.std(0)
    gi = int((mu.abs() / (sd + 1e-6)).argmax())                  # near-constant line: carries the offset
    BL[L] = (torch.tensor(wb).float(), float(bb), gi, float(mu[gi]), float(sd[gi]))
    bel = X @ BL[L][0] + bb
    lines.append(f"  layer {L}: belief line AUROC at forks (held out) {auroc(pb, OK[od_ & fork]):.2f}; offset line {gi} "
                 f"(mean {mu[gi]:+.3f}, spread {sd[gi]:.4f}); belief medians: " +
                 ", ".join(f"{k} {float(bel[torch.tensor(v)].median()) if v.any() else float('nan'):+.2f}" for k, v in groups.items()))

# ---- scout=1: where could a threshold sit? (Immy 2026-11-01: a sidechain compressor keyed only by belief that pulls the
# doubt crowd down once belief crosses a threshold.) Every word gets a held-out belief: the line is fit on the other half
# of the answers. Prints the spread per group and, per candidate threshold, how many right forks it catches vs how many
# ordinary words it would wrongly fire on. No edit is made.
if opt.get("scout") == "1":
    sg = dict(groups)
    sg["right fork he doubted after"] = groups["right fork, stated before"] & (np.array([l[4] for l in lab]) == "doubt")
    ordinary = [k for k in ("other thinking", "reply", "sentence start, no answer yet") if k in sg]
    ql = lambda b, v, ps: " ".join(f"{p}% {np.percentile(b[v], p):+.2f}" for p in ps) if v.any() else "-"
    keep = {}
    for L in RL:
        Xn = cap_all[L]["x"].numpy()
        tr = fork | NEU if NEUTRAL else fork
        y_ = np.where(fork, OK.astype(float), TH) if NEUTRAL else OK.astype(float)
        hb = np.zeros(len(ROLE))
        for fit, read in ((ev_, od_), (od_, ev_)):
            _, _, _, (wb, bb) = ridge(Xn[fit & tr], y_[fit & tr], Xn[:1])
            hb[read] = Xn[read] @ wb + bb
        keep[L] = hb
        lines.append(f"scout layer {L} (held-out belief):")
        for k in ("right fork, stated before", "right fork he doubted after", "right fork, first statement"):
            lines.append(f"  {k:30} {int(sg[k].sum()):5d}: {ql(hb, sg[k], (1, 5, 10, 25, 50))}")
        for k in ["wrong fork"] + ordinary:
            lines.append(f"  {k:30} {int(sg[k].sum()):5d}: {ql(hb, sg[k], (50, 90, 99, 99.9))}")
        anyord = np.zeros(len(ROLE), bool)
        for k in ordinary:
            anyord |= sg[k]
        for t in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            lines.append(f"  threshold {t:.2f}: catches {np.mean(hb[sg['right fork, stated before']] > t):5.1%} of right repeat forks, "
                         f"{np.mean(hb[sg['right fork he doubted after']] > t):5.1%} of the ones he doubted after, "
                         f"{np.mean(hb[sg['right fork, first statement']] > t):5.1%} of first statements; fires on "
                         f"{np.mean(hb[anyord] > t):6.2%} of ordinary words, {np.mean(hb[sg['wrong fork']] > t) if sg['wrong fork'].any() else float('nan'):5.1%} of wrong forks")
    # the doubt crowd: at right repeat forks, which neurons push toward doubt, and how much of the verdict they own.
    # "doubted" = he went on to doubt, "handed" = he handed in; the crowd's job is the difference between the two.
    kd = np.array([l[4] for l in lab])
    dz, hz = torch.tensor(sg["right fork he doubted after"]), torch.tensor(groups["right fork, stated before"] & (kd == "hand"))
    lines.append(f"crowd scout: verdict (hand-in minus doubt score) median {float(GAP[dz].median()):+.2f} where he doubted, "
                 f"{float(GAP[hz].median()):+.2f} where he handed in ({int(dz.sum())} / {int(hz.sum())} forks)")
    allv = []
    for L in RL:
        vd, vh = cap_all[L]["vote"][FK(dz)].float().mean(0), cap_all[L]["vote"][FK(hz)].float().mean(0)
        neg = vd < 0
        lines.append(f"  layer {L}: all neurons {float(vd.sum()):+.2f} where doubted / {float(vh.sum()):+.2f} where handed in; "
                     f"doubt voters {int(neg.sum())} carry {float(vd[neg].sum()):+.2f}; their doubted-minus-handed {float((vd - vh)[neg].sum()):+.2f}")
        allv += [(float(vd[j]), L, j) for j in torch.nonzero(neg).flatten().tolist()]
    allv.sort()
    tot = sum(v for v, _, _ in allv)
    run, marks = 0.0, {}
    for i, (v, _, _) in enumerate(allv):
        run += v
        for f in (0.5, 0.8, 0.95):
            if f not in marks and run <= f * tot:
                marks[f] = i + 1
    lines.append(f"  whole doubt crowd in layers {RL}: {len(allv)} neurons, {tot:+.2f} score points where he doubted; "
                 + ", ".join(f"{int(f * 100)}% of it in the top {marks.get(f, len(allv))}" for f in (0.5, 0.8, 0.95)))
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "scout.npz"), role=ROLE, ok=OK, prior=PRIOR, answer=R_,
             kind=np.array([l[4] for l in lab]), **{f"belief{L}": keep[L] for L in RL})
    open(os.path.join(out, "scout.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("SCOUT_DONE")
    sys.exit(0)

# ---- compressor=1 (Immy 2026-11-01): a sidechain compressor keyed only by belief. Layers must start at the gatekeeper
# layer (layers=17-23: gatekeeper 17, doubt crowd 18-23).
#   key        one quiet existing neuron in the gatekeeper layer is rewired: gate = steep * (belief - cth), up = belief,
#              so SiLU keeps it ~silent below the threshold and it fires above it
#   sidechain  it writes into one residual dimension that the rest of the model barely reads (the private wire)
#   ducking    the doubt crowd's gate rows read that wire: once it is up, each one is pushed to "off" (gate -shift),
#              so the crowd goes quiet only when the key is over threshold; belief itself never reads the wire
# options: cth=0.85 (threshold), steep=30, target=4 (score points of doubt vote to duck, picks how many neurons), amp=20
# (wire height at the forks, in units of that dimension's usual size)
COMP = opt.get("compressor") == "1"
if COMP:
    GK, CL = RL[0], RL[1:]
    CTH, STEEP, TARGET, AMP = float(opt.get("cth", 0.85)), float(opt.get("steep", 30)), float(opt.get("target", 4)), float(opt.get("amp", 20))
    kd = np.array([l[4] for l in lab])
    DZ = groups["right fork, stated before"] & (kd == "doubt"); HZ = groups["right fork, stated before"] & (kd == "hand")
    ORD = (ROLE == 0) | (ROLE == 2) | (ROLE == 3)
    groups["right fork he doubted after"], groups["right fork he handed in"] = DZ, HZ
    lines.append(f"compressor: key at layer {GK} (threshold {CTH}, steepness {STEEP}), crowd layers {CL}, target {TARGET} points, wire height {AMP}x")

    def resid():
        """the residual stream after the gatekeeper layer (input of the next layer) at every scored word"""
        out_, hs_ = [], []
        with torch.no_grad():
            for ids, pos in SEQ:
                cap = {}                                               # the key layer's output = the next layer's input
                h = lay[GK].register_forward_hook(
                    lambda mod, a, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, pos].float()))
                m.model(input_ids=torch.tensor([ids], device=DEV))
                h.remove()
                out_.append(cap["h"].cpu())
        return torch.cat(out_)

    # the private wire: the residual dimension that is small and that layers after the key barely read
    H0 = resid()
    size = H0.pow(2).mean(0).sqrt()                                   # usual size of each dimension
    readc = torch.zeros(H0.shape[1])
    with torch.no_grad():
        for L in range(GK + 1, len(lay)):
            nw = getattr(lay[L], ARCH["pre"]).weight.float().cpu()
            for nm in ("gate", "up"):
                readc += WT(L, nm).float().pow(2).sum(0).cpu() * nw ** 2
            ow = getattr(lay[L], ARCH["opnorm"]).weight.float().cpu()
            for nm_, mm in lay[L].named_modules():                     # attention / conv reads of the stream
                if (isinstance(mm, torch.nn.Linear) and mm.in_features == H0.shape[1] and not nm_.startswith(ARCH["mlp"])
                        and not nm_.endswith(("o_proj", "out_proj"))):
                    readc += mm.weight.float().pow(2).sum(0).cpu() * ow ** 2
        readc += m.lm_head.weight.float().pow(2).sum(0).cpu() * FINAL.weight.float().cpu() ** 2
    cost = readc * size ** 2                                          # how loud a wire of AMP x its size would be to the rest
    WIRE = int(cost.argmin())
    lines.append(f"  private wire: residual dimension {WIRE} (usual size {float(size[WIRE]):.3f}, read cost {float(cost[WIRE]):.3g}; "
                 f"median dimension {float(cost.median()):.3g}, rank {int((cost < cost[WIRE]).sum())} of {len(cost)})")

    # the key: the quietest neuron in the gatekeeper layer (its 99.9% write size over every scored word)
    with torch.no_grad():
        xg = cap_all[GK]["x"].to(DEV)
        actg = ARCH["act"](xg @ WT(GK, "gate").float().T) * (xg @ WT(GK, "up").float().T)
        wsz = WT(GK, "down").float().norm(dim=0)
        loud = torch.quantile((actg.abs() * wsz[None]).float(), .999, dim=0)
        GN = int(loud.argmin())
        del xg, actg
    lines.append(f"  key neuron: layer {GK} #{GN} (its old 99.9% write {float(loud[GN]):.2e}, median neuron {float(loud.median()):.2e})")
    EDIT = {}                                                          # (kind, layer, index) -> (orig, new)

    def put(kind, L, j, new):
        W_ = WT(L, kind)
        cur = W_[:, j] if kind == "down" else W_[j]
        EDIT[(kind, L, j)] = (EDIT.get((kind, L, j), (cur.clone(),))[0], new.to(W_.dtype).to(W_.device))

    def pam(on):
        with torch.no_grad():
            for (kind, L, j), (o, n) in EDIT.items():
                W_ = WT(L, kind)
                if kind == "down":
                    W_[:, j] = n if on else o
                else:
                    W_[j] = n if on else o

    def ridge0(X, y, alphas=(1, 10, 100, 1e3, 1e4)):
        """ridge with no intercept: these models have no constant input to carry an offset, and the one near-constant
        line (c1) wanders +-40%, which made the key misfire. The fit finds its own offset carriers across the input."""
        X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
        sc = np.sqrt((X ** 2).mean(0)) + 1e-6
        Z = X / sc
        folds = np.arange(len(y)) % 5
        best = None
        for al in alphas:
            err = 0.0
            for f in range(5):
                a, b_ = folds != f, folds == f
                w = np.linalg.solve(Z[a].T @ Z[a] + al * np.eye(Z.shape[1]), Z[a].T @ y[a])
                err += ((Z[b_] @ w - y[b_]) ** 2).sum()
            if best is None or err < best[0]:
                best = (err, al)
        w = np.linalg.solve(Z.T @ Z + best[1] * np.eye(Z.shape[1]), Z.T @ y)
        return torch.tensor(w / sc).float()

    Xg = cap_all[GK]["x"].numpy()
    trg = fork | NEU
    yg = np.where(fork, OK.astype(float), TH)
    up_row = ridge0(Xg[trg], yg[trg])
    gate_row = STEEP * ridge0(Xg[trg], yg[trg] - CTH)
    # placebo=1 (control): the same build with the belief labels shuffled over the same words, so the key reads no belief.
    # Its threshold is moved until it fires on as many words as the real key, and everything the real build sets at the
    # doubted-after forks (wire height, the crowd's "key at full" reading) is set where the fake key fires instead.
    # The ducked crowd stays the real one (picked by its doubt vote), so only WHEN the ducking happens changes.
    PLACEBO = opt.get("placebo") == "1"
    if PLACEBO:
        xall, real_gate = cap_all[GK]["x"], gate_row
        rate = float(((xall @ gate_row) > 0).float().mean())
        yp = yg.copy(); yp[trg] = np.random.default_rng(0).permutation(yg[trg])
        up_row = ridge0(Xg[trg], yp[trg])
        lo_, hi_ = -1.0, 2.0                                           # bisect the threshold to the real firing rate
        for _ in range(30):
            c_ = (lo_ + hi_) / 2
            gate_row = STEEP * ridge0(Xg[trg], yp[trg] - c_)
            r_ = float(((xall @ gate_row) > 0).float().mean())
            lo_, hi_ = (c_, hi_) if r_ > rate else (lo_, c_)
            if abs(r_ / rate - 1) < 0.02:
                break
        # A shuffled-label fit has almost no spread, so most words sat just under the threshold, in SiLU's dip (~-0.27),
        # and the fake key wrote a steady negative onto the wire (try 1, 2026-11-02: ordinary words -3.6x, which also
        # pushed the crowd UP everywhere). Stretch the gate to the real gate's spread: same firing words, but quiet
        # words sit as far below the threshold as the real key's do.
        zr, zp = xall @ real_gate, xall @ gate_row
        gate_row = gate_row * float(zr.std() / zp.std())
        lines.append(f"  PLACEBO gate stretched x{float(zr.std() / zp.std()):.2f}: median gate input real {float(zr.median()):+.1f}, "
                     f"placebo {float((xall @ gate_row).median()):+.1f}")
        FIRE = ((xall @ gate_row) > 0).numpy()
        lines.append(f"  PLACEBO key: labels shuffled, threshold {c_:+.3f}; fires on {r_:.2%} of words (real key {rate:.2%}); "
                     f"on {FIRE[DZ].mean():.0%} of doubted-after forks (real key fires on its own reference set)")
        DZ_REAL, DZ = DZ, FIRE                                         # the fake key's firing words stand in for the forks
    bel0 = cap_all[GK]["x"] @ up_row
    lines.append(f"  key reads belief with no offset line: ordinary words {float(bel0[torch.tensor(ORD)].median()):+.2f} "
                 f"(1% {float(bel0[torch.tensor(ORD)].quantile(.01)):+.2f}, 99% {float(bel0[torch.tensor(ORD)].quantile(.99)):+.2f}), "
                 f"doubted-after forks {float(bel0[torch.tensor(DZ)].median()):+.2f}")
    pw = POSTW(GK)
    down = torch.zeros(WT(GK, "down").shape[0]); down[WIRE] = 1.0 / float(pw[WIRE])
    put("gate", GK, GN, gate_row); put("up", GK, GN, up_row); put("down", GK, GN, down)
    pam(True)
    H1 = resid()
    dz_, hz_, od_w = torch.tensor(DZ), torch.tensor(HZ), torch.tensor(ORD)
    # scale the write to the asked wire height. Not in one step: the sandwich norm caps one neuron's share, so the rise
    # is concave in the write (2026-11-01: the one-step scale asked 8x / 4x / 2x and got 8.2 / 8.0 / 7.5, deep in the cap).
    # s <- s * asked / measured converges from above on a concave rise through zero.
    s_ = 1.0
    for it in range(10):
        rise = (H1[:, WIRE] - H0[:, WIRE]) / float(size[WIRE])
        got = float(rise[dz_].median())
        lines.append(f"  wire height step {it}: write x{s_:.4g}, rise at doubted-after forks {got:+.2f} (asked {AMP:g})")
        print(lines[-1], flush=True)
        if abs(got / AMP - 1) < 0.03:
            break
        s_ *= AMP / max(got, 1e-6)
        put("down", GK, GN, down * s_)
        pam(True)
        H1 = resid()
    rise = (H1[:, WIRE] - H0[:, WIRE]) / float(size[WIRE])
    lines.append(f"  wire rise (x usual size): doubted-after forks median {float(rise[dz_].median()):+.1f}, handed-in forks "
                 f"{float(rise[hz_].median()):+.1f}, ordinary words median {float(rise[od_w].median()):+.2f} "
                 f"(99% {float(rise[od_w].quantile(.99)):+.2f}, 99.9% {float(rise[od_w].quantile(.999)):+.2f}, min {float(rise[od_w].min()):+.2f})")
    st_ = torch.tensor(ROLE == 3)
    top = torch.argsort(-rise * st_.float())[:8].tolist()
    lines.append("  where the key fires before any answer is written (top 8 sentence starts, for reading):")
    flat = [(n, int(p_)) for n, (ids, pos) in enumerate(SEQ) for p_ in pos.tolist()]
    for w in top:
        n, p_ = flat[w]
        lines.append(f"    rise {float(rise[w]):+.1f}: ...{tok.decode(SEQ[n][0][max(0, p_ - 40):p_ + 1])!r}")

    # the ducking: the crowd reads the wire at its own input; pick the strongest doubt voters until they hold TARGET points
    # the key's firing at every word (layer 17's input is not changed by the edit, so it comes straight from the capture)
    with torch.no_grad():
        xg = cap_all[GK]["x"].to(DEV)
        KEY = (ARCH["act"](xg @ WT(GK, "gate")[GN].float()) * (xg @ WT(GK, "up")[GN].float())).cpu()
        del xg
    KHOT = float(KEY[dz_].median())
    groups["ordinary, key silent"] = ORD & (KEY < 0.1 * KHOT).numpy()
    groups["ordinary, key firing"] = ORD & (KEY >= 0.5 * KHOT).numpy()
    cap2 = {L: [] for L in CL}
    with torch.no_grad():
        for ids, pos in SEQ:
            hs = [MLP(L).register_forward_pre_hook(lambda mod, a, L=L: cap2[L].append(a[0][0, pos].float().cpu())) for L in CL]
            m.model(input_ids=torch.tensor([ids], device=DEV))
            for h in hs:
                h.remove()
    cand = []
    cz_ = torch.tensor(DZ_REAL) if PLACEBO else dz_                   # the crowd is always picked at the real forks
    for L in CL:
        v = cap_all[L]["vote"][FK(cz_)].float().mean(0)
        ghot = cap_all[L]["g"][FK(cz_)].float().median(0).values
        for j in torch.argsort(v)[:2000].tolist():
            if ghot[j] > 0 and v[j] < 0:
                cand.append((float(v[j]), L, j, float(ghot[j])))
    cand.sort()
    pick, held = [], 0.0
    for c in cand:
        if held <= -TARGET:
            break
        pick.append(c); held += c[0]
    # each crowd layer learns its sidechain read: a line at its own input that reads 1 when the key is at its usual fork
    # height and 0 when silent (fit on half the answers, checked on the other half, then refit on all)
    RD, WIN = {}, {}
    kt = (KEY / KHOT).numpy()
    for L in CL:
        X_ = torch.cat(cap2[L]).numpy()
        rh = ridge0(X_[ev_], kt[ev_])
        rd = torch.tensor(X_[od_]) @ rh
        o_, h_ = torch.tensor(ORD[od_]) & (KEY[torch.tensor(od_)] < 0.1 * KHOT), torch.tensor(DZ[od_])
        RD[L] = ridge0(X_, kt)
        WIN[L] = (float(rd[o_].median()), float(rd[o_].abs().quantile(.99)), float(rd[h_].median()))
        lines.append(f"  layer {L}: sidechain read (held out) silent words median {WIN[L][0]:+.3f}, 99% |read| {WIN[L][1]:.3f}; "
                     f"doubted-after forks {WIN[L][2]:+.2f}; ducking {sum(1 for c in pick if c[1] == L)} neurons")
        del X_
    for v, L, j, gh in pick:
        beta = gh + SHIFT                                              # read 1 at the fork -> gate pushed to -SHIFT
        row = WT(L, "gate")[j].float().clone() - beta * RD[L].to(DEV)
        put("gate", L, j, row)
    lines.append(f"  crowd ducked: {len(pick)} neurons holding {held:+.2f} points of doubt vote at doubted-after forks")
    placed = [{"layer": L, "neuron": j, "vote": round(v, 4), "gate_at_hot": round(gh, 2)} for v, L, j, gh in pick]
    DROP = set()
    NEW = {(L, j): None for _, L, j, _ in pick}
    print("\n".join(lines), flush=True)

# ---- 3. the doubt receptors: strongest doubt votes at right forks where he had stated it before, and engaged there
if not COMP:
    hot = torch.tensor(groups["right fork, stated before"])
    cand = []
    for L in RL:
        v = cap_all[L]["vote"][FK(hot)].float().mean(0)
        ghot = cap_all[L]["g"][FK(hot)].float().median(0).values
        for j in torch.argsort(v)[:200].tolist():                   # most negative = votes doubt
            if ghot[j] > 0:
                cand.append((float(v[j]), L, j, float(ghot[j])))
    cand = sorted(cand)[:K]
    DROP = {tuple(int(v) for v in x.split(":")) for x in opt["drop"].split(",")} if opt.get("drop") else set()
    cand = [c for c in cand if (c[1], c[2]) not in DROP]      # left untouched (no replacements pulled in)
    if DROP:
        lines.append(f"dropped (left untouched): {sorted(DROP)}")
    lines.append(f"doubt receptors ({len(cand)}): their summed vote at right repeat forks {sum(c[0] for c in cand):+.2f} score points "
                 f"(all neurons in layers {RL}: {float(sum(cap_all[L]['vote'][FK(hot)].float().mean(0).sum() for L in RL)):+.2f})")

    # ---- 4. the edit: shift each receptor's gate by -beta * (belief - theta)
    placed, ORIG, NEW = [], {}, {}
    with torch.no_grad():
        for v, L, j, gh in cand:
            wb, bb, gi, Gm, _ = BL[L]
            bel = cap_all[L]["x"] @ wb + bb
            bhot = float(bel[hot].median())
            beta = (gh + SHIFT) / max(bhot - TH, 0.1)
            add = -beta * wb.clone()
            add[gi] += -beta * (bb - TH) / Gm
            row = WT(L, "gate")
            ORIG[(L, j)] = row[j].clone()
            NEW[(L, j)] = (row[j].float() + add.to(row.device)).to(row.dtype)
            placed.append({"layer": L, "neuron": j, "vote": round(v, 3), "gate_at_hot": round(gh, 2), "beta": round(beta, 2)})
            lines.append(f"  layer {L} neuron {j:5d}: doubt vote {v:+.3f}, gate at right repeat forks {gh:+.2f}, beta {beta:.2f}")


    def pam(on):
        """Edit the receptors in place (on) or put the untouched rows back (off): one copy of the model, exact restore."""
        with torch.no_grad():
            for (L, j), r in (NEW if on else ORIG).items():
                WT(L, "gate")[j] = r


# ---- 5. paired runs: untouched vs PAM, same words
HI = torch.tensor(HAND_IDS, device=DEV); DI = torch.tensor(DOUBT_IDS, device=DEV)
gap0, gap1, kl = [], [], []
with torch.no_grad():
    for ids, pos in SEQ:
        t = torch.tensor([ids], device=DEV)
        pam(False)
        a = torch.log_softmax(m(input_ids=t).logits[0, pos.to(DEV)].float(), -1)
        pam(True)
        b = torch.log_softmax(m(input_ids=t).logits[0, pos.to(DEV)].float(), -1)
        gap0.append((torch.logsumexp(a[:, HI], -1) - torch.logsumexp(a[:, DI], -1)).cpu())
        gap1.append((torch.logsumexp(b[:, HI], -1) - torch.logsumexp(b[:, DI], -1)).cpu())
        kl.append((a.exp() * (a - b)).sum(-1).cpu())
gap0, gap1, kl = torch.cat(gap0), torch.cat(gap1), torch.cat(kl)
lines.append("paired (PAM minus untouched): hand-in minus doubt gap moved, and drift (KL, untouched to PAM)")
for k, v in groups.items():
    v = torch.tensor(v)
    if not v.any():
        lines.append(f"  {k:28}     0 words")
        continue
    d = gap1[v] - gap0[v]
    lines.append(f"  {k:28} {int(v.sum()):5d} words: gap {float(d.median()):+.2f} (95% {float(d.quantile(.95)):+.2f}); "
                 f"KL mean {float(kl[v].mean()):.4f}, 99% {float(kl[v].quantile(.99)):.4f}, max {float(kl[v].max()):.3f}")
print("\n".join(lines), flush=True)

pam(True)
os.makedirs(out, exist_ok=True)
if MODEL in ("lfm3", "qwen3"):
    m.save_pretrained(out)
    tok.save_pretrained(out)
    for fn in os.listdir(src):
        if fn.endswith(".py") or fn == "chat_template.jinja":
            shutil.copy(os.path.join(src, fn), out)
else:                                                      # the master file has every tower: patch only the changed matrices
    # Every matrix the build touched: the receptors' gate rows (NEW) and, for the compressor, the key's gate, up and down
    # (EDIT). 2026-11-01: this patched gate rows only, so the E4B compressor ce1 shipped without its trigger.
    from safetensors import safe_open
    from safetensors.torch import save_file
    key = lambda L, kind: f"model.language_model.layers.{L}.{ARCH['mlp']}.{ARCH[kind]}.weight"
    touched = {("gate", L) for L, _ in NEW} | ({(kind, L) for kind, L, _ in EDIT} if COMP else set())
    patch = {key(L, kind): WT(L, kind).detach().cpu().contiguous() for kind, L in sorted(touched)}
    print("saving changed matrices:", sorted(patch), flush=True)
    save_file(patch, os.path.join(out, "pam_rows.safetensors"))
    full = {}
    with safe_open(os.path.join(src, "model.safetensors"), "pt") as f:
        meta = f.metadata()
        for k_ in f.keys():
            full[k_] = patch.get(k_, None) if k_ in patch else f.get_tensor(k_)
    save_file(full, os.path.join(out, "model.safetensors"), metadata=meta)
    del full
    with safe_open(os.path.join(out, "model.safetensors"), "pt") as f:     # read back: the file must hold every change
        bad = [k_ for k_ in patch if not torch.equal(f.get_tensor(k_), patch[k_])]
    assert not bad, f"saved file is missing changes in {bad}"
    print(f"saved file checked: {len(patch)} changed matrices match", flush=True)
    for fn in os.listdir(src):
        if fn != "model.safetensors" and os.path.isfile(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), out)
json.dump({"model": MODEL, "layers": RL, "k": K, "shift": SHIFT, "theta": TH, "anchor": opt.get("anchor", ""), "claims": opt.get("claims", "phrase"), "drop": sorted(DROP),
           "hand_ids": HAND_IDS, "doubt_ids": DOUBT_IDS, "receptors": placed,
           **({"compressor": {"placebo": PLACEBO, "key_layer": GK, "key_neuron": GN, "threshold": CTH, "steep": STEEP, "wire": WIRE, "target": TARGET,
                              "amp": AMP, "sidechain_read_heldout": {str(L): WIN[L] for L in CL}}} if COMP else {})}, open(os.path.join(out, "pam.json"), "w"), indent=1)
open(os.path.join(out, "pam_report.txt"), "w").write("\n".join(lines) + "\n")
print("PAM_DONE")
