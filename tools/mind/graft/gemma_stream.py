#!/usr/bin/env python3
"""Gemma 4 E4B's text brain, run one room at a time straight from its safetensors file, so the whole model never has
to sit in memory. Follows the fork's src/models/gemma4.cpp: embeddings scaled by sqrt(2560), per-layer token inputs,
sliding rooms (head 256, rope 10k) and long-sight rooms (head 512, rope 1M on the first 64 pairs only), rooms 24-41
reusing the keys and values of rooms 22 (sliding) and 23 (long-sight), soft-capped logits.

Sequences must be at most 512 tokens (the sliding window), so every room is plainly causal.

  python gemma_stream.py <model.safetensors> <tokenizer.json> "prompt"    greedy next 12 tokens, to check against llama.cpp
"""
import math, sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

D, N_LAYER, PER = 2560, 42, 256
HEADS, KV = 8, 2
LONG = {5, 11, 17, 23, 29, 35, 41}                          # long-sight rooms
KV_FROM = 24                                                 # rooms from here reuse earlier keys and values
P = "model.language_model."


def rms(x, w=None, eps=1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def rope(x, theta, n_rot_pairs=None):
    """neox-style rotary embedding; x: [batch, heads, seq, dim]. Only the first n_rot_pairs frequency pairs turn
    (Gemma 4's long-sight rooms leave the rest still, via frequency factors of 1e30)"""
    d = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    if n_rot_pairs is not None:
        inv[n_rot_pairs:] = 0.0
    ang = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Gemma:
    def __init__(self, path, device="cpu"):
        self.f = safe_open(path, "pt")
        self.dev = device

    def t(self, name):
        return self.f.get_tensor(P + name).to(self.dev, torch.float32)

    def rows(self, name, ids):
        sl = self.f.get_slice(P + name)
        return torch.stack([sl[int(i)] for i in ids.flatten()]).to(self.dev, torch.float32).view(*ids.shape, -1)

    def room(self, il):
        p = f"layers.{il}."
        names = ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm",
                 "self_attn.q_norm", "self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                 "mlp.down_proj", "per_layer_input_gate", "per_layer_projection", "post_per_layer_input_norm"]
        w = {n: self.t(p + n + ".weight") for n in names}
        if il < KV_FROM:
            for n in ("self_attn.k_proj", "self_attn.v_proj", "self_attn.k_norm"):
                w[n] = self.t(p + n + ".weight")
        w["scalar"] = self.t(p + "layer_scalar")
        return w

    def run(self, ids, hook=None, ablate=None, no_per_layer=()):
        """ids: [batch, seq] Gemma token ids -> logits. hook(il, attention [b, heads, q, k]) sees every room's
        attention; ablate: {room: [heads]} silences those heads' outputs; no_per_layer: rooms that run without their
        per-layer token input (as a transplanted room must)"""
        B, T = ids.shape
        x = self.rows("embed_tokens.weight", ids) * math.sqrt(D)
        per = self.rows("embed_tokens_per_layer.weight", ids).view(B, T, N_LAYER, PER) * math.sqrt(PER)
        proj = (x @ self.t("per_layer_model_projection.weight").T / math.sqrt(D)).view(B, T, N_LAYER, PER)
        per = (rms(proj, self.t("per_layer_projection_norm.weight")) + per) / math.sqrt(2.0)
        shared = {}
        mask = torch.full((T, T), float("-inf"), device=self.dev).triu(1)
        for il in range(N_LAYER):
            w = self.room(il)
            long = il in LONG
            hd = 512 if long else 256
            h = rms(x, w["input_layernorm"])
            q = rms((h @ w["self_attn.q_proj"].T).view(B, T, HEADS, hd), w["self_attn.q_norm"]).transpose(1, 2)
            if il < KV_FROM:
                k = rms((h @ w["self_attn.k_proj"].T).view(B, T, KV, hd), w["self_attn.k_norm"]).transpose(1, 2)
                v = rms((h @ w["self_attn.v_proj"].T).view(B, T, KV, hd)).transpose(1, 2)
                k = rope(k, 1e6, 64) if long else rope(k, 1e4)
                if il in (22, 23):
                    shared[long] = (k, v)
            else:
                k, v = shared[long]
            q = rope(q, 1e6, 64) if long else rope(q, 1e4)
            k, v = k.repeat_interleave(HEADS // KV, 1), v.repeat_interleave(HEADS // KV, 1)
            att = torch.softmax(q @ k.transpose(-1, -2) + mask, -1)          # attention scale 1 in Gemma 4
            if hook:
                hook(il, att)
            o = att @ v                                                       # [b, heads, seq, hd]
            if ablate and il in ablate:
                o[:, ablate[il]] = 0.0
            x = x + rms(o.transpose(1, 2).reshape(B, T, -1) @ w["self_attn.o_proj"].T, w["post_attention_layernorm"])
            h = rms(x, w["pre_feedforward_layernorm"])
            m = F.gelu(h @ w["mlp.gate_proj"].T, approximate="tanh") * (h @ w["mlp.up_proj"].T)
            x = x + rms(m @ w["mlp.down_proj"].T, w["post_feedforward_layernorm"])
            if il not in no_per_layer:
                g = F.gelu(x @ w["per_layer_input_gate"].T, approximate="tanh") * per[:, :, il]
                x = x + rms(g @ w["per_layer_projection"].T, w["post_per_layer_input_norm"])
            x = x * w["scalar"]
            del w
        x = rms(x, self.t("norm.weight"))
        logits = x @ self.t("embed_tokens.weight").T if not hasattr(self, "_emb") else x @ self._emb.T
        return 30.0 * torch.tanh(logits / 30.0)


if __name__ == "__main__":
    from tokenizers import Tokenizer
    g = Gemma(sys.argv[1])
    tok = Tokenizer.from_file(sys.argv[2])
    ids = [2] + tok.encode(sys.argv[3], add_special_tokens=False).ids                 # <bos> first, as llama.cpp does
    out = []
    with torch.no_grad():
        for _ in range(12):
            logits = g.run(torch.tensor([ids]))
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            ids.append(nxt)
            print(repr(tok.decode(out)), flush=True)
