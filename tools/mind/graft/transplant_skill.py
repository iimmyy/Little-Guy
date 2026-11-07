#!/usr/bin/env python3
"""Transplant test 2: graft the tissue that holds a known skill, and test that skill.

find_skill.py located Gemma E4B's copying circuit ("this came before, so what followed it last time comes next"):
a previous-word head in room 4 (head 2) feeding a copying head in room 5 (head 7); silencing the top copying heads
broke copying 4-8x while random heads did nothing, and rooms 4-5 copy as well without their per-layer token input.

The host and training are as in transplant_test.py (same text, steps and order per seed). Arms:
  graft:4,5     rooms 4 and 5, healing        - the tissue with the skill
  scratch:4,5   the same shapes from random   - same size, no skill
  host_only     no extra block
  graft:8,9     rooms 8 and 9, healing        - Gemma tissue without this skill (the first test's graft)
The skill test: random host-vocabulary sequences shown twice; loss on the second copy (copyable) against the first
(not guessable). Ordinary held-out text is measured too, so a skill bought with worse language would show.

usage: transplant_skill.py <gemma safetensors> <data folder> <out folder> <steps> <seeds, e.g. 0,1,2> [arms...]
"""
import json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transplant_test as T
from gemma_stream import LONG, rms, rope

ARMS = ["graft:4,5", "scratch:4,5", "host_only", "graft:8,9"]
COPY_L = 50


class Room(torch.nn.Module):
    """one Gemma 4 room with its own keys and values, sliding or long-sight, without the per-layer token input"""

    def __init__(self, long):
        super().__init__()
        self.long, hd = long, (512 if long else 256)
        self.hd = hd
        L = lambda i, o: torch.nn.Linear(i, o, bias=False)
        self.q, self.k, self.v, self.o = L(T.G_D, T.G_HEADS * hd), L(T.G_D, T.G_KV * hd), L(T.G_D, T.G_KV * hd), L(T.G_HEADS * hd, T.G_D)
        self.gate, self.up, self.down = L(T.G_D, T.G_FF), L(T.G_D, T.G_FF), L(T.G_FF, T.G_D)
        P = lambda n: torch.nn.Parameter(torch.ones(n))
        self.attn_norm, self.attn_post, self.ffn_norm, self.ffn_post = P(T.G_D), P(T.G_D), P(T.G_D), P(T.G_D)
        self.q_norm, self.k_norm, self.scalar = P(hd), P(hd), P(1)
        self.keep_att, self.att = False, None                # look inside: keep this room's attention pattern

    def load(self, f, il):
        p = f"model.language_model.layers.{il}."
        t = lambda n: f.get_tensor(p + n).float()
        with torch.no_grad():
            for mod, n in ((self.q, "self_attn.q_proj"), (self.k, "self_attn.k_proj"), (self.v, "self_attn.v_proj"),
                           (self.o, "self_attn.o_proj"), (self.gate, "mlp.gate_proj"), (self.up, "mlp.up_proj"),
                           (self.down, "mlp.down_proj")):
                mod.weight.copy_(t(n + ".weight"))
            for par, n in ((self.attn_norm, "input_layernorm"), (self.attn_post, "post_attention_layernorm"),
                           (self.ffn_norm, "pre_feedforward_layernorm"), (self.ffn_post, "post_feedforward_layernorm"),
                           (self.q_norm, "self_attn.q_norm"), (self.k_norm, "self_attn.k_norm")):
                par.copy_(t(n + ".weight"))
            self.scalar.copy_(t("layer_scalar"))
        return self

    def forward(self, x):
        B, Tn, _ = x.shape
        h = rms(x, self.attn_norm)
        q = rms(self.q(h).view(B, Tn, T.G_HEADS, self.hd), self.q_norm).transpose(1, 2)
        k = rms(self.k(h).view(B, Tn, T.G_KV, self.hd), self.k_norm).transpose(1, 2)
        v = rms(self.v(h).view(B, Tn, T.G_KV, self.hd)).transpose(1, 2)
        q, k = (rope(q, 1e6, 64), rope(k, 1e6, 64)) if self.long else (rope(q, 1e4), rope(k, 1e4))
        k, v = k.repeat_interleave(T.G_HEADS // T.G_KV, 1), v.repeat_interleave(T.G_HEADS // T.G_KV, 1)
        if self.keep_att:
            mask = torch.full((Tn, Tn), float("-inf"), device=x.device).triu(1)
            att = torch.softmax(q @ k.transpose(-1, -2) + mask, -1)
            self.att = att.detach()
            a = att @ v
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        x = x + rms(self.o(a.transpose(1, 2).reshape(B, Tn, -1)), self.attn_post)
        h = rms(x, self.ffn_norm)
        x = x + rms(self.down(F.gelu(self.gate(h), approximate="tanh") * self.up(h)), self.ffn_post)
        return x * self.scalar


def main():
    donor, data, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    seeds = [int(s) for s in sys.argv[5].split(",")]
    arms = sys.argv[6:] or ARMS
    os.makedirs(out, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = Tokenizer.from_file(os.path.join(data, "smollm2-tokenizer.json"))
    S, B = T.SEQ, T.BATCH
    train = T.tokens(os.path.join(data, "wt103-train-00000-of-00002.parquet"), tok, limit=steps * B * (S + 1) + S)
    val = T.tokens(os.path.join(data, "wt103-validation-00000-of-00001.parquet"), tok)
    vstarts = np.random.default_rng(1).integers(0, len(val) - S - 1, 64)
    vx = torch.tensor(np.stack([val[s:s + S + 1] for s in vstarts]), dtype=torch.long, device=dev)
    R = torch.tensor(np.random.default_rng(7).integers(1000, 20000, size=(64, COPY_L)))
    cx = torch.cat([R, R], 1).to(dev)                         # the skill test, never seen in training

    qpos = torch.arange(COPY_L + 1, 2 * COPY_L - 1)              # second-copy positions; target = where the next word sat
    tpos = qpos - COPY_L + 1

    def evaluate(model):
        model.eval()
        rooms = list(model.graft.layers) if model.graft is not None and model.graft.on else []
        with torch.no_grad():
            text = float(np.mean([F.cross_entropy(model(vx[i:i + 16, :-1]).flatten(0, 1), vx[i:i + 16, 1:].flatten()).item()
                                  for i in range(0, len(vx), 16)]))
            for r in rooms:
                r.keep_att = True
            lp = F.cross_entropy(model(cx[:, :-1]).flatten(0, 1), cx[:, 1:].flatten(), reduction="none").view(len(cx), -1)
            inside = {}
            for i, r in enumerate(rooms):
                r.keep_att = False
                a = r.att[:, :, qpos, :]
                copy = a[:, :, torch.arange(len(qpos)), tpos].mean((0, 2))        # per head: on the word that came next
                prev = a[:, :, torch.arange(len(qpos)), qpos - 1].mean((0, 2))    # per head: on the word just before
                inside[f"block_room{i}"] = dict(copy=[round(float(c), 3) for c in copy],
                                                previous=[round(float(c), 3) for c in prev])
                r.att = None
        model.train()
        return dict(text=text, first=float(lp[:, :COPY_L - 1].mean()), second=float(lp[:, COPY_L:].mean()), inside=inside)

    results = json.load(open(os.path.join(out, "skill.json"))) if os.path.exists(os.path.join(out, "skill.json")) else {}
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(train) // (S + 1))[: steps * B]
        for arm in arms:
            key = f"{arm}|seed{seed}"
            if key in results:
                continue
            torch.manual_seed(100 * seed)
            graft = None
            if arm != "host_only":
                kind, rooms = arm.split(":")
                rooms = [int(r) for r in rooms.split(",")]
                layers = [Room(r in LONG) for r in rooms]
                if kind == "graft":
                    with safe_open(donor, "pt") as f:
                        layers = [l.load(f, r) for l, r in zip(layers, rooms)]
                else:
                    for l in layers:
                        for m in l.modules():
                            if isinstance(m, torch.nn.Linear):
                                torch.nn.init.normal_(m.weight, std=0.02)
                torch.manual_seed(100 * seed + 1)
                graft = T.Graft(layers)
            torch.manual_seed(100 * seed + 2)
            model = T.Host(tok.get_vocab_size()).to(dev)
            if graft is not None:
                model.graft = graft.to(dev)
            groups = [dict(params=[p for n, p in model.named_parameters() if not n.startswith("graft.")], lr=T.LR)]
            if graft is not None:
                groups.append(dict(params=list(graft.inp.parameters()) + list(graft.out.parameters()), lr=T.LR))
                groups.append(dict(params=list(graft.layers.parameters()), lr=T.GRAFT_LR if arm.startswith("graft") else T.LR))
            opt = torch.optim.AdamW(groups, weight_decay=0.0)
            base = [g["lr"] for g in opt.param_groups]
            curve, t0 = [], time.time()
            for s in range(steps):
                w = min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps))
                for g, b in zip(opt.param_groups, base):
                    g["lr"] = b * w
                idx = order[s * B:(s + 1) * B]
                x = torch.tensor(np.stack([train[i * (S + 1):(i + 1) * (S + 1)] for i in idx]), dtype=torch.long, device=dev)
                loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if (s + 1) % 250 == 0 or s == 0 or s == steps - 1:
                    curve.append(dict(step=s + 1, seconds=round(time.time() - t0), **evaluate(model)))
                    c = curve[-1]
                    print(f"{key:20s} step {c['step']:5d}  text {c['text']:.3f}  copy: first {c['first']:.2f} "
                          f"second {c['second']:.2f}  {c['seconds']}s", flush=True)
                    for name, v in c["inside"].items():
                        print(f"{'':20s}   inside {name}: best copier head {int(np.argmax(v['copy']))} {max(v['copy']):.3f}"
                              f" (all {v['copy']}), best tagger head {int(np.argmax(v['previous']))} {max(v['previous']):.3f}",
                              flush=True)
            r = dict(curve=curve, final=curve[-1])
            if graft is not None:
                graft.on = False
                r["block_off"] = evaluate(model)
                graft.on = True
                print(f"{key:20s} block off: text {r['block_off']['text']:.3f}  second copy {r['block_off']['second']:.2f}",
                      flush=True)
            results[key] = r
            if graft is not None:
                torch.save(model.state_dict(), os.path.join(out, f"model-{arm.replace(':', '').replace(',', '')}-seed{seed}.pt"))
            json.dump(results, open(os.path.join(out, "skill.json"), "w"), indent=1)
            del model, graft, opt
            if dev.type == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
