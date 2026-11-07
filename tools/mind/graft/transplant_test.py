#!/usr/bin/env python3
"""The transplant test: does a piece of a big model work when grafted into a small one?

A small host language model learns English. Halfway up it gets a graft: two of Gemma E4B's layers (rooms 8 and 9),
with a small translator on each side where the graft meets the host (model stitching), behind a gate that starts
closed: the output translator starts at zero (as LoRA's second matrix does), so the graft changes nothing at first
and is used only as far as it helps. (A tanh gate at zero, as in Flamingo, opened too slowly in the smoke test: 0.009
after 100 steps, which left every arm's block unused.)

Four arms, same host start, same text, same order, same steps:
  graft_heal    Gemma's layers, healing (trained at a tenth of the host's learning rate)
  graft_frozen  Gemma's layers, untouched: only the translators and the host learn
  scratch       the same two layers, same shapes, started from random numbers, trained at the host's rate
  host_only     no extra block at all
The test passes if graft_heal learns faster and ends better than scratch. At the end each arm's block is also switched
off, to see how much the host came to rely on it.

The graft runs Gemma's layer as in the fork's gemma4.cpp, minus the per-layer token input (it is keyed to Gemma's
own vocabulary, which the host does not use).

usage: transplant_test.py <gemma safetensors> <data folder> <out folder> <steps> [arms...]
"""
import json, math, os, sys, time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

SEQ, BATCH = 256, 8
HOST_D, HOST_LAYERS, HOST_HEADS, GRAFT_AT = 512, 6, 8, 3
DONOR_LAYERS = [8, 9]                                        # two early rooms with their own keys (not KV-shared)
G_D, G_HEADS, G_KV, G_HEAD, G_FF, G_THETA = 2560, 8, 2, 256, 10240, 10000.0
LR, GRAFT_LR = 1e-3, 1e-4


def rms(x, w=None, eps=1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def rope(x, theta):
    """rotate-half rotary position embedding over the last dim; x: [batch, heads, seq, dim]"""
    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device).float() / d)
    ang = torch.arange(x.shape[2], device=x.device).float()[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class GemmaLayer(torch.nn.Module):
    """one Gemma 4 sliding-attention layer (the window is longer than our sequences, so it is plain causal)"""

    def __init__(self):
        super().__init__()
        L = lambda i, o: torch.nn.Linear(i, o, bias=False)
        self.q, self.k, self.v, self.o = L(G_D, G_HEADS * G_HEAD), L(G_D, G_KV * G_HEAD), L(G_D, G_KV * G_HEAD), L(G_HEADS * G_HEAD, G_D)
        self.gate, self.up, self.down = L(G_D, G_FF), L(G_D, G_FF), L(G_FF, G_D)
        P = lambda n: torch.nn.Parameter(torch.ones(n))
        self.attn_norm, self.attn_post, self.ffn_norm, self.ffn_post = P(G_D), P(G_D), P(G_D), P(G_D)
        self.q_norm, self.k_norm, self.scalar = P(G_HEAD), P(G_HEAD), P(1)

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
        B, T, _ = x.shape
        h = rms(x, self.attn_norm)
        q = rms(self.q(h).view(B, T, G_HEADS, G_HEAD), self.q_norm).transpose(1, 2)
        k = rms(self.k(h).view(B, T, G_KV, G_HEAD), self.k_norm).transpose(1, 2)
        v = rms(self.v(h).view(B, T, G_KV, G_HEAD)).transpose(1, 2)
        q, k = rope(q, G_THETA), rope(k, G_THETA)
        k, v = k.repeat_interleave(G_HEADS // G_KV, 1), v.repeat_interleave(G_HEADS // G_KV, 1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)   # Gemma 4: attention scale 1
        x = x + rms(self.o(a.transpose(1, 2).reshape(B, T, -1)), self.attn_post)
        h = rms(x, self.ffn_norm)
        x = x + rms(self.down(F.gelu(self.gate(h), approximate="tanh") * self.up(h)), self.ffn_post)
        return x * self.scalar


class HostLayer(torch.nn.Module):
    def __init__(self, d=HOST_D, heads=HOST_HEADS):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = torch.nn.Parameter(torch.ones(d)), torch.nn.Parameter(torch.ones(d))
        self.qkv, self.o = torch.nn.Linear(d, 3 * d, bias=False), torch.nn.Linear(d, d, bias=False)
        self.up, self.down = torch.nn.Linear(d, 4 * d, bias=False), torch.nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(rms(x, self.n1)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q, 10000.0), rope(k, 10000.0), v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.gelu(self.up(rms(x, self.n2))))


class Graft(torch.nn.Module):
    """translator in -> the transplanted layers -> translator out, which starts at zero"""

    def __init__(self, layers):
        super().__init__()
        self.inp = torch.nn.Linear(HOST_D, G_D, bias=False)
        self.layers = torch.nn.ModuleList(layers)
        self.out = torch.nn.Linear(G_D, HOST_D, bias=False)
        torch.nn.init.zeros_(self.out.weight)
        self.on = True

    def forward(self, h):
        if not self.on:
            return h
        x = self.inp(h)
        for layer in self.layers:
            x = layer(x)
        return h + self.out(x)


class Host(torch.nn.Module):
    def __init__(self, vocab, graft=None):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, HOST_D)
        self.layers = torch.nn.ModuleList(HostLayer() for _ in range(HOST_LAYERS))
        self.norm = torch.nn.Parameter(torch.ones(HOST_D))
        self.graft = graft
        torch.nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ids):
        h = self.emb(ids)
        for i, layer in enumerate(self.layers):
            if i == GRAFT_AT and self.graft is not None:
                h = self.graft(h)
            h = layer(h)
        return rms(h, self.norm) @ self.emb.weight.T


def tokens(path, tok, limit=None):
    text = [t for t in pq.read_table(path).column("text").to_pylist() if t.strip()]
    ids = []
    for i in range(0, len(text), 2000):
        for e in tok.encode_batch(text[i:i + 2000]):
            ids.extend(e.ids)
        if limit and len(ids) >= limit:
            break
    return np.asarray(ids[:limit] if limit else ids, np.int32)


def main():
    donor, data, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    arms = sys.argv[5:] or ["graft_heal", "graft_frozen", "scratch", "host_only"]
    os.makedirs(out, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = Tokenizer.from_file(os.path.join(data, "smollm2-tokenizer.json"))
    need = steps * BATCH * (SEQ + 1)
    train = tokens(os.path.join(data, "wt103-train-00000-of-00002.parquet"), tok, limit=need + SEQ)
    val = tokens(os.path.join(data, "wt103-validation-00000-of-00001.parquet"), tok)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(train) // (SEQ + 1))[: steps * BATCH]       # every arm reads the same text, in the same order
    vrng = np.random.default_rng(1)
    vstarts = vrng.integers(0, len(val) - SEQ - 1, 64)
    vx = torch.tensor(np.stack([val[s:s + SEQ + 1] for s in vstarts]), dtype=torch.long, device=dev)
    print(f"{len(train):,} training tokens ({steps} steps x {BATCH} x {SEQ}), {len(val):,} held-out; device {dev}", flush=True)

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            losses = [F.cross_entropy(model(vx[i:i + 16, :-1]).flatten(0, 1), vx[i:i + 16, 1:].flatten()).item()
                      for i in range(0, len(vx), 16)]
        model.train()
        return float(np.mean(losses))

    results = {}
    for arm in arms:
        torch.manual_seed(0)
        graft = None
        if arm != "host_only":
            layers = [GemmaLayer() for _ in DONOR_LAYERS]
            if arm.startswith("graft"):
                with safe_open(donor, "pt") as f:
                    layers = [l.load(f, il) for l, il in zip(layers, DONOR_LAYERS)]
            else:
                for l in layers:                             # a fresh block: small random weights, norms at 1
                    for m in l.modules():
                        if isinstance(m, torch.nn.Linear):
                            torch.nn.init.normal_(m.weight, std=0.02)
            torch.manual_seed(1)
            graft = Graft(layers)
        torch.manual_seed(2)
        model = Host(tok.get_vocab_size()).to(dev)           # the same host start in every arm
        if graft is not None:
            model.graft = graft.to(dev)
        groups = [dict(params=[p for n, p in model.named_parameters() if not n.startswith("graft.")], lr=LR)]
        if graft is not None:
            groups.append(dict(params=list(graft.inp.parameters()) + list(graft.out.parameters()), lr=LR))
            block = list(graft.layers.parameters())
            if arm == "graft_frozen":
                for p in block:
                    p.requires_grad_(False)
            else:
                groups.append(dict(params=block, lr=GRAFT_LR if arm == "graft_heal" else LR))
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
        base = [g["lr"] for g in opt.param_groups]
        curve, t0 = [], time.time()
        for s in range(steps):
            w = min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps))   # warm up, then ease off
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * w
            idx = order[s * BATCH:(s + 1) * BATCH]
            x = torch.tensor(np.stack([train[i * (SEQ + 1):(i + 1) * (SEQ + 1)] for i in idx]), dtype=torch.long, device=dev)
            loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            if (s + 1) % 100 == 0 or s == 0:
                curve.append(dict(step=s + 1, train=float(loss.detach()), val=evaluate(model), seconds=round(time.time() - t0, 1),
                                  out_size=float(graft.out.weight.norm()) if graft is not None else None))
                c = curve[-1]
                print(f"{arm:13s} step {c['step']:5d}  train {c['train']:.3f}  held-out {c['val']:.3f}"
                      + (f"  out-translator {c['out_size']:.3f}" if graft is not None else "") + f"  {c['seconds']:.0f}s", flush=True)
        r = dict(curve=curve, final=curve[-1]["val"])
        if graft is not None:
            graft.on = False
            r["block_off"] = evaluate(model)                 # how much the host came to rely on the block
            graft.on = True
            print(f"{arm:13s} block switched off: held-out {r['block_off']:.3f} (with it {r['final']:.3f})", flush=True)
        results[arm] = r
        json.dump(dict(donor_layers=DONOR_LAYERS, steps=steps, batch=BATCH, seq=SEQ, lr=LR, graft_lr=GRAFT_LR,
                       results=results), open(os.path.join(out, "transplant.json"), "w"), indent=1)
        del model, graft, opt
        if dev.type == "mps":
            torch.mps.empty_cache()
    print({a: round(r["final"], 3) for a, r in results.items()}, flush=True)


if __name__ == "__main__":
    main()
