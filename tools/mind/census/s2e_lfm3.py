#!/usr/bin/env python3
"""Steer2Edit (Sun et al., arXiv 2602.09870, NeurIPS 2026; github.com/Trustworthy-ML-Lab/Steer2Edit at 86382c2) ported
to LFM3 as the fair baseline for the belief compressor: same model, same night test, same engine.

Their efficient-reasoning recipe, kept as published (efficient_reasoning/extract_directions.py + edit.py):
  1. the model answers their 200 GSM8K probe questions (sampled, its own generation settings)
  2. direction per layer = mean MLP output over the response of the 5% shortest answers minus the 5% longest
     (also the mean MLP down-projection input, for the neuron scores)
  3. every MLP neuron i: score g_i = cos(direction, W_down[:, i] * mean input_i); elastic-net closed form
     lambda_i = sign(g_i) max(|g_i| - rho alpha, 0) / (rho (1 - alpha)); the edit adds lambda_i * sign(W_down[:, i] . v) * v
     to that neuron's down column (v = unit direction). Attention is left alone (their Qwen3-4B-Thinking setting).
Only the module names change (LFM3: feed_forward.w2). Their conv-free assumption does not matter: MLPs only.

usage: s2e_lfm3.py directions <hf model> <probe answers jsonl> <out.pt>     (answers from engine_run.py on the probe)
       s2e_lfm3.py edit <hf model> <directions.pt> <out folder> rho=0.7 alpha=0.1
"""
import json, os, shutil, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from patients import _lfm3

TAIL = "\n\nEnd with a line: Answer: <your answer>"   # the night test's prompt, so the probe answers look like test answers


def directions(src, ans, out, percent=0.05):
    p = _lfm3("cuda", src)
    tok, m = p.tok, p.model
    rows = [r for r in map(json.loads, open(ans)) if r["thinking"]]
    full = lambda r: "\n" + r["thinking"] + "</think>" + (r["reply"] or "")
    rows.sort(key=lambda r: r["think_tokens"])
    k = max(1, int(percent * len(rows)))
    short, long_ = rows[:k], rows[-k:]
    print(f"{len(rows)} probe answers, {k} shortest ({short[-1]['think_tokens']} words max) vs {k} longest "
          f"({long_[0]['think_tokens']} words min)", flush=True)
    lay = m.model.layers
    L = [i for i in range(len(lay)) if hasattr(lay[i], "feed_forward")]
    cap = {}
    hs = [lay[i].feed_forward.w2.register_forward_hook(
        lambda mod, x, y, i=i: cap.__setitem__(i, (x[0][0].float(), y[0].float()))) for i in L]

    def means(group):
        pre, post = {i: 0 for i in L}, {i: 0 for i in L}
        for r in group:
            pr = tok(tok.apply_chat_template([{"role": "user", "content": r["q"] + TAIL}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True), add_special_tokens=False).input_ids
            ids = (pr + tok(full(r), add_special_tokens=False).input_ids)[:4096]      # their right truncation
            with torch.no_grad():
                m(input_ids=torch.tensor([ids], device="cuda"))
            s = len(pr)
            for i in L:                                   # their span: response tokens, shifted by one as in their code
                pre[i] = pre[i] + cap[i][0][s - 1:len(ids) - 1].mean(0).cpu() / len(group)
                post[i] = post[i] + cap[i][1][s - 1:len(ids) - 1].mean(0).cpu() / len(group)
        return pre, post

    pp, qp = means(short)
    pn, qn = means(long_)
    for h in hs:
        h.remove()
    torch.save({"layers": L, "dir": {i: qp[i] - qn[i] for i in L}, "x_mean": {i: 0.5 * (pp[i] + pn[i]) for i in L},
                "meta": {"probe": ans, "k": k, "percent": percent}}, out)
    print("DIRECTIONS_DONE", out)


def edit(src, dpath, out, rho, alpha):
    D = torch.load(dpath)
    p = _lfm3("cuda", src)
    m = p.model
    lay = m.model.layers
    n_ed, per = 0, {}
    with torch.no_grad():
        for i in D["layers"]:
            Wd = lay[i].feed_forward.w2.weight
            W = Wd.data.float()
            v = F.normalize(D["dir"][i].cuda(), dim=0)
            G = F.cosine_similarity(v.unsqueeze(1), W * D["x_mean"][i].cuda().unsqueeze(0), dim=0)
            lam = torch.sign(G) * torch.clamp(G.abs() - rho * alpha, min=0) / (rho * (1 - alpha))
            ks = torch.sign(W.T @ v)
            lam = torch.where(ks != 0, lam * ks, torch.zeros_like(lam))
            idx = torch.nonzero(lam).flatten()
            W[:, idx] += v.unsqueeze(1) * lam[idx].unsqueeze(0)
            Wd.data.copy_(W.to(Wd.dtype))
            per[i] = int(len(idx)); n_ed += len(idx)
    print(f"rho {rho} alpha {alpha}: {n_ed} neurons edited; per layer {per}", flush=True)
    m.save_pretrained(out)
    p.tok.save_pretrained(out)
    for f in os.listdir(src):                             # chat template etc. the save does not carry
        if f.endswith((".jinja", ".json")) and not os.path.exists(os.path.join(out, f)):
            shutil.copy(os.path.join(src, f), out)
    json.dump({"steer2edit": {"rho_mlp": rho, "alpha": alpha, "neurons": n_ed, "per_layer": per, "directions": dpath}},
              open(os.path.join(out, "s2e.json"), "w"), indent=1)
    print("EDIT_DONE", out)


if __name__ == "__main__":
    mode = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    pos = [a for a in sys.argv[2:] if "=" not in a]
    if mode == "directions":
        directions(*pos)
    else:
        edit(*pos, float(opt.get("rho", 0.7)), float(opt.get("alpha", 0.1)))
