#!/usr/bin/env python3
"""Turn the song up with no word on the page. Reads only, no weights changed. GOAL-turnup-v1.md.
usage: turnup_read.py dirs <hf> <tok> <steps> <dirs.pt> <n>
       turnup_read.py scan <hf> <tok> <steps> <dirs.pt> <out> <skip> <n>
       turnup_read.py held <hf> <tok> <steps> <dirs.pt> <out> <word> <layer> <mode> [core]"""
import json, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
cmd, hf, tkp, steps, dirsf = sys.argv[1:6]
DEV, MAXTOK = "cuda", 7000
tok = AutoTokenizer.from_pretrained(tkp, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, trust_remote_code=True).to(DEV).eval().requires_grad_(False)
EXPR = re.compile(r"(?<![\d.,])\d[\d.,]*\s*(?:\*|×|\\times|\\cdot|/|\+|-)\s*\d[\d.,]*\s*=\s*$")
SEP = {"plain": ". ", "yes": ". Yes, ", "wait": ". Wait, "}
NL = len(m.model.layers)
ST = dict(L=None, v=None, mode="last", cap=None)
def mk(L):
    def f(mod, args, kwargs):
        hs = args[0]
        if ST["cap"] is not None:
            ST["cap"][L] = hs[0, -1].float().cpu()
        if ST["L"] == L and ST["v"] is not None:
            hs = hs.clone()
            if ST["mode"] == "last":
                hs[0, -1] = hs[0, -1] + ST["v"].to(hs.dtype)
            else:
                hs[0] = hs[0] + ST["v"].to(hs.dtype)
            return (hs,) + args[1:], kwargs
    return f
for i, l in enumerate(m.model.layers):
    l.register_forward_pre_hook(mk(i), with_kwargs=True)
def head(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=True) + "\n"
@torch.inference_mode()
def read(ids, a, b):
    l = m(input_ids=torch.tensor([ids], device=DEV), logits_to_keep=1).logits[0, -1].float()
    return float(l[a] - l[b])
def setup(j):
    h = head(j["q"]); written, right = j["broken"], j["clean"]
    expr = EXPR.search(j["pre"]).group()
    W = tok(h + j["pre"] + written, add_special_tokens=False).input_ids
    R = tok(h + j["pre"] + right, add_special_tokens=False).input_ids
    diff = [i for i, (x, y) in enumerate(zip(W, R)) if x != y]
    if len(W) != len(R) or len(diff) != 1 or len(W) > MAXTOK:
        return None
    q = diff[0]; a, b = R[q], W[q]
    L0 = len(tok(h + j["pre"], add_special_tokens=False).input_ids)
    shared = tok.decode(R[L0:q]) if q > L0 else ""
    ids = {}
    for name, sep in SEP.items():
        for page, num, ids0 in (("R", right, R), ("W", written, W)):
            x = tok(h + j["pre"] + num + sep + expr + shared, add_special_tokens=False).input_ids
            if x[:len(ids0)] != ids0:
                return None
            ids[f"{name}:{page}"] = x
    first = tok(h + j["pre"] + shared, add_special_tokens=False).input_ids
    return dict(a=a, b=b, ids=ids, first=first)
J = [json.loads(l) for l in open(steps)]
J = [j for j in J if "broken" in j and EXPR.search(j["pre"])]
BUILD = [j for j in J if j["split"] == "build"]
HELD = [j for j in J if j["split"] == "held"]
def by_problem(js):  # same problems stay together, order fixed by the file
    return sorted(js, key=lambda j: (hash(j["id"]) * 0 + sum(map(ord, str(j["id"]))) % 97, str(j["id"]), j["step"]))
BUILD = by_problem(BUILD)
t0 = time.time()
if cmd == "dirs":
    n = int(sys.argv[6]); sum_ = {w: torch.zeros(NL, m.config.hidden_size) for w in ("yes", "wait")}; c, used = 0, 0
    for j in BUILD:
        if used >= n:
            break
        s = setup(j)
        if s is None:
            continue
        for page in "RW":
            ST["cap"] = {}; read(s["ids"][f"plain:{page}"], s["a"], s["b"]); p = torch.stack([ST["cap"][i] for i in range(NL)])
            for w in ("yes", "wait"):
                ST["cap"] = {}; read(s["ids"][f"{w}:{page}"], s["a"], s["b"])
                sum_[w] += torch.stack([ST["cap"][i] for i in range(NL)]) - p
            c += 1
        used += 1
        if used % 10 == 0:
            print(used, round(time.time() - t0), flush=True)
    ST["cap"] = None
    d = {w: v / c for w, v in sum_.items()}
    torch.save(dict(d=d, used=used, last_id=str(j["id"])), dirsf)
    for w in d:
        print(w, "size by layer", [round(float(d[w][i].norm()), 2) for i in range(NL)], flush=True)
    print("DIRS_DONE", used, round(time.time() - t0), flush=True)
    sys.exit()
D = torch.load(dirsf)
dd = D["d"]
def K_C(s, word=None, L=None, mode="last", alpha=1.0, v=None):
    """m on right and wrong pages, plain text, knob on"""
    if word is not None or v is not None:
        ST.update(L=L, mode=mode, v=(alpha * (dd[word][L] if v is None else v)).to(DEV))
    else:
        ST.update(L=None, v=None)
    r = read(s["ids"]["plain:R"], s["a"], s["b"]); w = read(s["ids"]["plain:W"], s["a"], s["b"])
    ST.update(L=None, v=None)
    return r, w
if cmd == "scan":
    outf, skip, n = sys.argv[6], int(sys.argv[7]), int(sys.argv[8])
    LAYERS = list(range(1, NL, 2))
    used, seen = 0, 0
    with open(outf, "w") as w_:
        for j in BUILD:
            s = setup(j)
            if s is None:
                continue
            seen += 1
            if seen <= skip:
                continue
            if used >= n:
                break
            row = dict(step=j["step"], id=j["id"], plain=K_C(s), reads={})
            for w in ("yes", "wait"):
                for mode in ("last", "all"):
                    for L in LAYERS:
                        row["reads"][f"{w}:{mode}:{L}"] = K_C(s, w, L, mode)
            w_.write(json.dumps(row) + "\n"); w_.flush()
            used += 1
            print(used, n, round(time.time() - t0), flush=True)
    print("SCAN_DONE", used, round(time.time() - t0), flush=True)
    sys.exit()
if cmd == "held":
    outf, word, L, mode = sys.argv[6], sys.argv[7], int(sys.argv[8]), sys.argv[9]
    core = len(sys.argv) > 10 and sys.argv[10] == "core"
    g = torch.Generator().manual_seed(1105)
    rnd = torch.randn(m.config.hidden_size, generator=g); rnd = rnd / rnd.norm() * dd[word][L].norm()
    spk = dd["wait"][L] - dd["yes"][L]
    n, skipped = 0, 0
    with open(outf, "w") as w_:
        for j in HELD:
            s = setup(j)
            if s is None:
                skipped += 1; continue
            row = dict(step=j["step"], id=j["id"], reads={"plain": K_C(s), "knob:1": K_C(s, word, L, mode, 1.0)})
            if not core:
                row["reads"]["knob:0"] = K_C(s, word, L, mode, 0.0)
                row["reads"]["knob:0.5"] = K_C(s, word, L, mode, 0.5)
                row["reads"]["knob:2"] = K_C(s, word, L, mode, 2.0)
                row["reads"]["random:1"] = K_C(s, word, L, mode, 1.0, v=rnd)
                row["reads"]["speakers:1"] = K_C(s, word, L, mode, 1.0, v=spk)
                # first look, nothing on the page
                ST.update(L=None, v=None); row["m0"] = read(s["first"], s["a"], s["b"])
                ST.update(L=L, mode=mode, v=dd[word][L].to(DEV)); row["m0_knob"] = read(s["first"], s["a"], s["b"])
                # damage: top word on the last 200 positions of the page, knob on against off
                x = torch.tensor([s["ids"]["plain:W"]], device=DEV)
                with torch.inference_mode():
                    ST.update(L=None, v=None); t0_ = m(input_ids=x, logits_to_keep=200).logits[0].argmax(-1)
                    ST.update(L=L, mode=mode, v=dd[word][L].to(DEV)); t1_ = m(input_ids=x, logits_to_keep=200).logits[0].argmax(-1)
                ST.update(L=None, v=None)
                row["top_same"] = float((t0_ == t1_).float().mean())
            w_.write(json.dumps(row) + "\n"); w_.flush()
            n += 1
            if n % 10 == 0:
                print(n, len(HELD), round(time.time() - t0), flush=True)
    print("HELD_DONE", n, "skipped", skipped, round(time.time() - t0), flush=True)
