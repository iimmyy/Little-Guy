#!/usr/bin/env python3
"""Our E4B (the grid-stored brain, e4b-grid4) with its word ends stripped: the 42 rooms, without the dictionary that
turns words into codes, the per-layer word table, or the output that turns codes back into words.

What is kept is everything between the ends: the rooms, and the wiring that turns the entry code into each room's
per-layer input (per_layer_model_proj). The per-layer word table's slot stays open: Gemma adds a word-table part to
each room's per-layer input, and here a host can put its own part in that slot (per_extra), or leave it empty.

The file is only read, never written (it is a release brain). Rooms are stored as bf16 on the GPU; the stream, the
norms and attention run in float32. Math follows gemma_stream.py, which matches llama.cpp exactly.

  python gemma_body.py <e4b-grid4.gguf> <tokenizer.json> "prompt"
      puts the word ends back on and says the next 12 tokens greedily: proves the body is built right
"""
import math, sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from gguf import GGUFReader
from gguf.quants import dequantize

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from gemma_stream import D, N_LAYER, PER, HEADS, KV, LONG, KV_FROM, rms, rope

ROOM = {"attn_norm": "attn_norm", "attn_post": "post_attention_norm", "ffn_norm": "ffn_norm", "ffn_post": "post_ffw_norm",
        "q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "q_norm": "attn_q_norm", "k_norm": "attn_k_norm",
        "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down", "pl_gate": "inp_gate", "pl_proj": "proj",
        "pl_post": "post_norm", "scalar": "layer_output_scale"}


class File:
    def __init__(self, path):
        self.r = GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}

    def get(self, name, rows=None):
        """a tensor as float32 in torch's [out, in] layout; rows picks rows of a big table without unpacking it all"""
        t = self.t[name]
        data = t.data if rows is None else t.data[rows]
        a = dequantize(data, t.tensor_type) if t.tensor_type.name != "F32" else np.asarray(data, dtype=np.float32)
        shape = [int(s) for s in reversed(t.shape)]
        if rows is not None:
            shape[0] = len(rows)
        return torch.from_numpy(np.ascontiguousarray(a).reshape(shape))


class Body(torch.nn.Module):
    def __init__(self, path, device, rooms=range(N_LAYER), dtype=torch.bfloat16):
        super().__init__()
        f = File(path)
        self.dev, self.rooms = device, list(rooms)
        frozen = lambda x, big=True: torch.nn.Parameter(x.to(device, dtype if big else torch.float32), requires_grad=False)
        self.w = torch.nn.ModuleList()
        for il in self.rooms:
            m = torch.nn.Module()
            for key, n in ROOM.items():
                name = f"blk.{il}.{n}.weight"
                if name in f.t:
                    x = f.get(name)
                    setattr(m, key, frozen(x, big=x.dim() == 2))
            self.w.append(m)
        self.proj = frozen(f.get("per_layer_model_proj.weight"))
        self.proj_norm = frozen(f.get("per_layer_proj_norm.weight"), big=False)
        self.out_norm = frozen(f.get("output_norm.weight"), big=False)
        self.file = f
        self.recompute = True

    def scramble(self, seed):
        """the control: every matrix replaced by random numbers of the same size and spread, so Tokyo keeps its shape
        and scale but loses what training put in it (norm weights and room scalars are kept)"""
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() == 2:
                    std = float(p.float().std())
                    p.copy_((torch.randn(p.shape, generator=g) * std).to(p.device, p.dtype))
        return self

    @staticmethod
    def lin(x, w):
        return F.linear(x.to(w.dtype), w).float()

    def forward(self, x, per_extra=None, hook=None):
        """x: the entry code [batch, seq, 2560], at Gemma's scale. per_extra: [batch, seq, 42, 256] put in the word
        table's slot, or None. hook(il, attention) sees each room's attention. Returns the final normed code."""
        B, T, _ = x.shape
        per = rms(self.lin(x, self.proj).view(B, T, N_LAYER, PER) / math.sqrt(D), self.proj_norm)
        per = (per + per_extra) / math.sqrt(2.0) if per_extra is not None else per / math.sqrt(2.0)
        shared = {}
        recompute = self.recompute and torch.is_grad_enabled() and hook is None
        for i, il in enumerate(self.rooms):
            kv = shared.get(il in LONG, (None, None)) if il >= KV_FROM else (None, None)
            if recompute:      # keep only each room's input; redo the room on the way back (gradient checkpointing)
                x, k, v = checkpoint(self.room, i, x, per[:, :, il], *kv, use_reentrant=False)
            else:
                x, k, v = self.room(i, x, per[:, :, il], *kv, hook=hook)
            if il in (22, 23):
                shared[il in LONG] = (k, v)
        return rms(x, self.out_norm)

    def room(self, i, x, per, k=None, v=None, hook=None):
        il, w = self.rooms[i], self.w[i]
        B, T, _ = x.shape
        long = il in LONG
        hd = 512 if long else 256
        h = rms(x, w.attn_norm)
        q = rms(self.lin(h, w.q).view(B, T, HEADS, hd), w.q_norm).transpose(1, 2)
        if il < KV_FROM:
            k = rms(self.lin(h, w.k).view(B, T, KV, hd), w.k_norm).transpose(1, 2)
            v = rms(self.lin(h, w.v).view(B, T, KV, hd)).transpose(1, 2)
            k = rope(k, 1e6, 64) if long else rope(k, 1e4)
        q = rope(q, 1e6, 64) if long else rope(q, 1e4)
        kk, vv = k.repeat_interleave(HEADS // KV, 1), v.repeat_interleave(HEADS // KV, 1)
        if hook:
            mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
            att = torch.softmax(q @ kk.transpose(-1, -2) + mask, -1)
            hook(il, att)
            a = att @ vv
        else:
            a = F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=1.0)
        x = x + rms(self.lin(a.transpose(1, 2).reshape(B, T, -1), w.o), w.attn_post)
        h = rms(x, w.ffn_norm)
        x = x + rms(self.lin(F.gelu(self.lin(h, w.gate), approximate="tanh") * self.lin(h, w.up), w.down), w.ffn_post)
        g = F.gelu(self.lin(x, w.pl_gate), approximate="tanh") * per
        x = x + rms(self.lin(g, w.pl_proj), w.pl_post)
        return x * w.scalar, k, v


class WordEnds:
    """the stripped ends, only for proving the body is built right"""

    def __init__(self, body):
        self.b = body
        self.emb = body.file.get("token_embd.weight").to(body.dev, torch.bfloat16)

    def enter(self, ids):
        x = self.emb[ids].float() * math.sqrt(D)
        flat = ids.flatten().cpu().numpy()
        per = self.b.file.get("per_layer_token_embd.weight", rows=flat).to(self.b.dev)
        return x, per.view(*ids.shape, N_LAYER, PER) * math.sqrt(PER)

    def leave(self, h):
        return 30.0 * torch.tanh(Body.lin(h, self.emb) / 30.0)


if __name__ == "__main__":
    from tokenizers import Tokenizer
    import time
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    body = Body(sys.argv[1], dev)
    ends = WordEnds(body)
    print(f"body built in {time.time() - t0:.0f}s, {sum(p.numel() for p in body.parameters()) / 1e9:.2f}B numbers",
          flush=True)
    tok = Tokenizer.from_file(sys.argv[2])
    ids = [2] + tok.encode(sys.argv[3], add_special_tokens=False).ids
    out = []
    with torch.no_grad():
        for _ in range(12):
            x, per = ends.enter(torch.tensor([ids], device=dev))
            nxt = int(ends.leave(body(x, per))[0, -1].argmax())
            out.append(nxt)
            ids.append(nxt)
    print(repr(tok.decode(out)), flush=True)
