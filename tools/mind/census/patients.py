"""Patients for the dosing pipeline (Immy, 2026-10-28: dose other models "like a doctor looking at symptoms").

Every exam (giants.py, vitals.py, the PAM) needs the same few things from a model: its layers, the norms that read the
bloodstream (the receptor sites), the word that closes its thinking, and its final word scores. Each patient below
says where those live in one model, so a new model is one entry here, not a rewrite of every exam.

    from patients import load
    p = load("lfm3", dev="mps")
    p.model, p.tok, p.body(ids) -> hidden states, p.norms -> [(layer, kind, module)], p.end, p.logits(h)

kind is the operator a norm feeds: "attn" (attention), "conv" (LFM3's short convolution) or "mlp".
"""
import os
from dataclasses import dataclass
from typing import Callable

import torch

HF = os.path.expanduser("~/.littleguy/hf")


@dataclass
class Patient:
    name: str
    model: torch.nn.Module
    tok: object
    body: Callable            # input ids (1, T) on dev -> last hidden states (1, T, D), final norm applied
    norms: list               # [(layer index, kind, norm module)] in the order the stream meets them
    end: int                  # token id that closes the thinking
    logits: Callable          # last hidden states (T, D) -> word scores (T, V), float
    layer_types: list


def _e4b(dev):
    from e4b_mac import load as e4b_load
    model, tok = e4b_load(dev)
    lay = model.model.layers
    cap = model.config.final_logit_softcapping

    def logits(h):
        lg = model.lm_head(h).float()
        return cap * torch.tanh(lg / cap) if cap else lg
    norms = [(i, "attn", l.input_layernorm) for i, l in enumerate(lay)] + \
            [(i, "mlp", l.pre_feedforward_layernorm) for i, l in enumerate(lay)]
    return Patient("e4b", model, tok, lambda ids: model.model(input_ids=ids).last_hidden_state, norms,
                   tok.convert_tokens_to_ids("<channel|>"), logits, ["attn"] * len(lay))


def _lfm3(dev, path=None):
    path = path or os.environ.get("LFM3_PATH", HF + "/lfm3-1.7b")    # LFM3_PATH: a changed copy (bake_budget.py)
    # Liquid's own model code (read in full 2026-10-28: plain torch, no I/O); transformers 5.21 has no native lfm3
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).to(dev)
    model.eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    lay, types = model.model.layers, model.config.layer_types
    kind = {"conv": "conv", "full_attention": "attn", "sliding_attention": "attn"}
    norms = [(i, kind[types[i]], l.operator_norm) for i, l in enumerate(lay)] + \
            [(i, "mlp", l.ffn_norm) for i, l in enumerate(lay)]
    return Patient("lfm3", model, tok, lambda ids: model.model(input_ids=ids).last_hidden_state, norms,
                   tok.convert_tokens_to_ids("</think>"), lambda h: model.lm_head(h).float(), list(types))


PATIENTS = {"e4b": _e4b, "lfm3": _lfm3}


def load(name, dev="mps"):
    return PATIENTS[name](dev)
