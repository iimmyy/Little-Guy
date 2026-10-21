#!/usr/bin/env python3
"""The PyTorch side of engine/ears-check: what his ears should give for a sound, to compare with llama.cpp.
Runs on CPU (only the ears and their embedder are loaded, not the brain).

  feat       MODEL SOUND48.f32 OUT_DIR        old.npy [T,128] from the 16 kHz version (transformers' extractor),
                                              hi.npy [T,39] (widen.py HighBands), sound16.f32
  embd       MODEL SOUND48.f32 OUT.npy [SIDE.pt|random]   what the ears hand the brain [tokens, n_embd]; with SIDE.pt
                                              the side path is on; "random" = random non-zero side weights (saved
                                              next to OUT as side-random.pt), so any mistake in a port shows;
                                              a sep SIDE.pt (the high end on its own page) puts its tokens after
                                              his, one per `pool` of his
SOUND48.f32 = raw mono float32 at 48 kHz. MODEL = the transformers checkpoint folder (with the audio tower).
"""
import os, sys

import numpy as np
import torch
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import widen as W


def ears(model_dir):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder
    cfg = AutoConfig.from_pretrained(model_dir)
    tower = Gemma4AudioModel(cfg.audio_config).float().eval()
    emb = Gemma4MultimodalEmbedder(cfg.audio_config, cfg.text_config).float().eval()
    sd_t, sd_e = {}, {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("model.audio_tower."):
                sd_t[k[len("model.audio_tower."):]] = f.get_tensor(k).float()
            if k.startswith("model.embed_audio."):
                sd_e[k[len("model.embed_audio."):]] = f.get_tensor(k).float()
    tower.load_state_dict(sd_t)
    emb.load_state_dict(sd_e)
    return cfg, tower, emb


def features(model_dir, x48):
    from transformers import AutoFeatureExtractor
    y16 = resample_poly(x48, 1, 3).astype(np.float32)
    f = AutoFeatureExtractor.from_pretrained(model_dir)([y16], sampling_rate=16000, return_tensors="pt",
                                                        padding="longest")
    return y16, f


def feat(model_dir, sound, out_dir):
    x = np.fromfile(sound, np.float32)
    y16, f = features(model_dir, x)
    T = int(f["input_features_mask"][0].sum())
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "old.npy"), f["input_features"][0, :T].numpy())
    np.save(os.path.join(out_dir, "hi.npy"), W.HighBands()(torch.from_numpy(x), T).numpy())
    y16.tofile(os.path.join(out_dir, "sound16.f32"))
    print({"frames": T})


def embd(model_dir, sound, out, side_pt=None):
    cfg, tower, emb = ears(model_dir)
    x = np.fromfile(sound, np.float32)
    _, f = features(model_dir, x)
    mask = f["input_features_mask"]
    ck = torch.load(side_pt, map_location="cpu") if side_pt and side_pt != "random" else {}
    page = None
    if ck.get("where") == "sep":
        side = W.Side(tower.subsample_conv_projection, cfg.audio_config.output_proj_dims, ck.get("mid", 0), bias=True)
        side.load_state_dict(ck["side"])
        hi = W.HighBands()(torch.from_numpy(x), f["input_features"].shape[1]).unsqueeze(0) * mask[..., None]
        n = ((int(mask[0].sum()) - 1) // 2) // 2 + 1
        with torch.no_grad():
            page = emb(inputs_embeds=W.pool_time(side(hi, mask), n, ck.get("pool", 4)))
        side_pt = None                                        # his own tokens are untouched below
    if side_pt:
        proj = tower.subsample_conv_projection
        side = W.Side(proj, cfg.audio_config.hidden_size)
        if side_pt == "random":
            torch.manual_seed(0)
            with torch.no_grad():
                side.out.weight.normal_(0, 0.05)
                for conv, norm in (side.l0, side.l1):
                    conv.weight.mul_(1.3); norm.weight.add_(0.1)
            torch.save({"side": side.state_dict(), "edges_hz": W.HighBands().edges.tolist(), "step": "random"},
                       os.path.join(os.path.dirname(os.path.abspath(out)), "side-random.pt"))
        else:
            side.load_state_dict(torch.load(side_pt, map_location="cpu")["side"])
        hi = W.HighBands()(torch.from_numpy(x), f["input_features"].shape[1]).unsqueeze(0) * mask[..., None]
        old = proj.forward
        proj.forward = lambda feats, m=None: (lambda h, mm: (h + side(hi, m)[:, :h.shape[1]], mm))(*old(feats, m))
    with torch.no_grad():
        out_e = emb(inputs_embeds=tower(f["input_features"].float(), mask).last_hidden_state)
    n = ((int(mask[0].sum()) - 1) // 2) // 2 + 1
    res = out_e[0, :n]
    if page is not None:
        res = torch.cat([res, page[0]])
    np.save(out, res.numpy())
    print({"tokens": n, "page_tokens": 0 if page is None else page.shape[1], "side": side_pt or ck.get("where")})


if __name__ == "__main__":
    {"feat": feat, "embd": embd}[sys.argv[1]](*sys.argv[2:])
