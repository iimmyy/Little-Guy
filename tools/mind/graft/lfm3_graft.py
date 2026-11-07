#!/usr/bin/env python3
"""Transplant 3: E4B's common-sense rooms grafted into LFM3 (Immy, 2026-10-28: "take the things from E4B that are
clearly working for it and that LFM is missing ... a Frankenstein model, best of both worlds").

The diagnosis (census/rooms.py) found the tissue: knocking out E4B rooms 10-13 (and 8-15 around them) breaks the
HellaSwag questions E4B gets right and LFM3 gets wrong far more than equally sure questions both get right, and that
skill leans on the rooms' per-layer word notes. The socket (stitch_site.py): LFM3's stream after its layer 4 is the
one a straight line turns best into what room 8 reads (same relative depth).

The graft, after LFM3 layer SITE:   h  ->  h + B( rooms(A h) - A h )
  A (2048 -> 2560) translates LFM3's stream into E4B's, started from the least-squares map;
  rooms = E4B's own rooms, frozen, fed their NATIVE word notes: each LFM3 piece looks its note up through its E4B word
  twin (twins.py), the common denominator, so the rooms read the input they were trained on (transplant 2's lesson:
  without its native input the copying head never woke up);
  B (2560 -> 2048) translates what the rooms wrote back, started at zero, so the graft changes nothing at first.
LFM3 stays frozen; only A and B learn (stage 1: does the organ work at all, before any healing).

Teaching signal on WikiText: LFM3's own next-piece loss plus E4B's next-piece odds at the spots where both word lists
end a piece at the same letter, compared over the pieces both lists share (cross-tokenizer distillation on exact
matches; ULD 2024, DSKD 2024 and approximate likelihood matching 2025 handle the rest more fully).

Arms (same text, same order, same steps, same trainable size):
  graft     E4B's rooms
  scratch   the same rooms re-drawn at random, frozen: is it E4B's tissue or just the translators and extra depth?
  teacher   no rooms: a 2048 -> 2560 -> 2048 adapter at the same spot, same teacher: is it the tissue or the teaching?
  host      nothing trained: LFM3 as released
Exams: held-out WikiText loss, drift from the released LFM3 (KL), HellaSwag 1000 (and the 163 gap questions).

usage: lfm3_graft.py <arm> <out folder> [rooms=8-15 site=4 steps=1000 bs=8 seq=512 lr=1e-4 kd=1 seed=0 hs=1000]
"""
import json, math, os, random, sys, time
from collections import UserDict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../census"))
from patients import load

arm, out = sys.argv[1:3]
opt = dict(a.split("=", 1) for a in sys.argv[3:])
r0, r1 = map(int, opt.get("rooms", "8-15").split("-"))
ROOMS, SITE = list(range(r0, r1 + 1)), int(opt.get("site", 4))
STEPS, BS, SEQ = int(opt.get("steps", 1000)), int(opt.get("bs", 8)), int(opt.get("seq", 512))
LR, KD, SEED, NHS = float(opt.get("lr", 1e-4)), float(opt.get("kd", 1)), int(opt.get("seed", 0)), int(opt.get("hs", 1000))
ACC = int(opt.get("acc", 1))                                         # each batch run in ACC parts (memory), same update
LEASH, TEACH, TEXT = float(opt.get("leash", 0)), opt.get("teach", "all"), opt.get("text", "wt103")
DEV = "cuda"
HERE = os.path.dirname(__file__)
DATA = opt.get("data", os.path.join(HERE, "../../../data"))
BENCH = os.path.join(HERE, "../../../.run/bench")
os.makedirs(out, exist_ok=True)
torch.manual_seed(SEED); random.seed(SEED)
log = open(f"{out}/log.txt", "a")


def say(s):
    print(s, flush=True); log.write(s + "\n"); log.flush()


say(f"=== {arm} rooms {ROOMS} site {SITE} steps {STEPS} bs {BS} seq {SEQ} lr {LR} kd {KD} teach {TEACH} "
    f"leash {LEASH} text {TEXT} seed {SEED}")
lfm, e4b = load("lfm3", DEV), load("e4b", DEV)
tl, te = lfm.tok, e4b.tok
lm = e4b.model.model
twin = torch.load(f"{DATA}/twins.pt")
TW = twin["e4b_id"].to(DEV)                                         # LFM3 piece -> its E4B twin's number
SH = twin["exact"].nonzero()[:, 0].to(DEV)                          # LFM3 pieces with an exact twin
SH = SH[SH < lfm.model.lm_head.weight.shape[0]]
SH_E = TW[SH]


# ---------------------------------------------------------------- text, read by both word lists, paired by letter
def windows(path, n_chars, n_win):
    t = open(path).read()[:n_chars]
    enc = tl(t, add_special_tokens=False, return_offsets_mapping=True)
    off, ids, ws = enc.offset_mapping, enc.input_ids, []
    for i in range(0, len(ids) - SEQ, SEQ - 1):
        s0, s1 = off[i][0], off[i + SEQ - 2][1]
        span = t[s0:s1]
        b = tl(span, add_special_tokens=False, return_offsets_mapping=True)
        a = te(span, add_special_tokens=False, return_offsets_mapping=True)
        b_ids, a_ids = [tl.bos_token_id] + b.input_ids[:SEQ - 1], [te.bos_token_id] + a.input_ids[:int(SEQ * 1.3)]
        ao, bo = a.offset_mapping[:len(a_ids) - 1], b.offset_mapping[:len(b_ids) - 1]
        end_a = {e: j + 1 for j, (_, e) in enumerate(ao)}
        # (LFM3 spot, E4B spot, same next piece): both have read up to the same letter, and if both next pieces end
        # at the same letter too, both predict the same piece of text and their chances of it can be compared
        pairs = [(i2 + 1, end_a[e], i2 + 1 < len(bo) and end_a[e] < len(ao) and bo[i2 + 1][1] == ao[end_a[e]][1])
                 for i2, (_, e) in enumerate(bo) if e in end_a]
        ws.append((b_ids, a_ids, pairs))
        if len(ws) >= n_win:
            break
    return ws


def pad(rows, v):
    T = max(map(len, rows))
    return torch.tensor([r + [v] * (T - len(r)) for r in rows], device=DEV), \
        torch.tensor([[1] * len(r) + [0] * (T - len(r)) for r in rows], device=DEV)


FILES = {"wt103": ("wt103/wt103-train-00000-of-00002.txt", "wt103/wt103-validation-00000-of-00001.txt"),
         "c4": ("c4/train.txt", "c4/held.txt")}                    # c4: everyday web text, prepared by c4_text.py
train = windows(f"{DATA}/{FILES[TEXT][0]}", 12_000_000, STEPS * BS)
held = windows(f"{DATA}/{FILES[TEXT][1]}", 10 ** 9, 64)
say(f"text: {len(train)} training windows (repeat as needed), {len(held)} held-out; "
    f"paired spots {sum(len(w[2]) for w in held) / sum(len(w[0]) - 1 for w in held):.1%}")


# ---------------------------------------------------------------- the graft
class Graft(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.A = torch.nn.Linear(2048, 2560)
        self.B = torch.nn.Linear(2560, 2048)
        torch.nn.init.zeros_(self.B.weight); torch.nn.init.zeros_(self.B.bias)
        if kind == "scratch":                                       # same shapes, random numbers, frozen
            import copy
            self.rooms = torch.nn.ModuleList(copy.deepcopy(lm.layers[r]) for r in ROOMS)
            for m in self.rooms.modules():                          # every matrix re-drawn; norm scales kept
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, std=lm.config.initializer_range)
            self.rooms.requires_grad_(False)
        elif kind == "graft":
            self.rooms = [lm.layers[r] for r in ROOMS]              # E4B's own, frozen, shared with the teacher
        self.ids = None

    def notes(self, ids):
        e = TW[ids]
        emb = lm.embed_tokens(e)
        return lm.project_per_layer_inputs(emb, lm.get_per_layer_inputs(e, emb))

    def forward(self, h):
        x = self.A(h.float()).to(h.dtype)
        if self.kind == "teacher":                                  # adapter, no rooms
            return h + self.B(F.gelu(x.float())).to(h.dtype)
        T = h.shape[1]
        pos = torch.arange(T, device=DEV)[None]
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        mk = dict(config=lm.config, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=pos)
        mask = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
        pe = {t: lm.rotary_emb(x, pos, t) for t in lm.unique_layer_types}
        ple = self.notes(self.ids)
        g = x
        for r, room in zip(ROOMS, self.rooms):
            t = lm.config.layer_types[r]
            g = room(g, ple[:, :, r], shared_kv_states=UserDict(), position_embeddings=pe[t], attention_mask=mask[t],
                     position_ids=pos, past_key_values=None)
        return h + self.B((g - x).float()).to(h.dtype)


graft = None
if arm != "host":
    graft = Graft(arm).to(DEV)
    if arm != "teacher" and os.path.exists(f"{DATA}/site_map_room{ROOMS[0]}_lfm{SITE}.pt"):
        m = torch.load(f"{DATA}/site_map_room{ROOMS[0]}_lfm{SITE}.pt")   # least-squares start for A
        with torch.no_grad():
            graft.A.weight.copy_(m["W"].T.float()); graft.A.bias.copy_((m["my"] - m["mx"] @ m["W"]).float())
        say("A started from the least-squares site map")
hook_off = [False]                                                  # the drift exam switches the graft off
if graft is not None:
    lfm.model.model.layers[SITE].register_forward_hook(lambda mod, a, o: o if hook_off[0] else graft(o))


def student(ids):
    if graft is not None:
        graft.ids = ids
    return lfm.model(input_ids=ids).logits.float()


def losses(batch, train_mode):
    """LFM3's own loss, E4B's teaching and the leash. teach=all: E4B's odds at every paired spot; teach=sel: only where
    E4B gave the real next piece a better chance than the released LFM3 did (learn from the expert where it is right)"""
    b_ids, b_m = pad([w[0] for w in batch], tl.pad_token_id)
    lp = student(b_ids).log_softmax(-1)                     # (word scores dropped at once: 125K words x 4K spots)
    ce = F.nll_loss(lp[:, :-1].flatten(0, 1), b_ids[:, 1:].flatten(), reduction="none")
    ce = (ce * b_m[:, 1:].flatten()).sum() / b_m[:, 1:].sum()
    zero = ce * 0
    need_orig = graft is not None and (LEASH > 0 or TEACH == "sel")
    if need_orig:
        with torch.no_grad():
            hook_off[0] = True
            lp0 = lfm.model(input_ids=b_ids).logits.float().log_softmax(-1)
            hook_off[0] = False
    leash = zero
    if graft is not None and LEASH > 0:
        kl = (lp0.exp() * (lp0 - lp)).sum(-1)
        leash = (kl * b_m).sum() / b_m.sum()
    if KD == 0 and train_mode:
        return ce, zero, leash, 0.0
    a_ids, _ = pad([w[1] for w in batch], te.pad_token_id)
    si, sj, bi, same = [], [], [], []
    for k, w in enumerate(batch):
        for i, j, sm in w[2]:
            si.append(i); sj.append(j); bi.append(k); same.append(sm)
    bi, si, sj, same = (torch.tensor(v, device=DEV) for v in (bi, si, sj, same))
    na = a_ids[bi, (sj + 1).clamp(max=a_ids.shape[1] - 1)]
    with torch.no_grad():                                   # E4B's word scores only where needed, 512 spots at a time
        hid = lm(input_ids=a_ids).last_hidden_state[bi, sj]   # (262K words x 4K spots at once is 4 GB: out of memory)
        t_sh, p_e4b = [], []
        for c in range(0, len(hid), 512):
            z = e4b.logits(hid[c:c + 512])
            t_sh.append(z[:, SH_E]); p_e4b.append(z.gather(-1, na[c:c + 512, None])[:, 0] - z.logsumexp(-1))
        t_sh, p_e4b = torch.cat(t_sh), torch.cat(p_e4b)       # p_e4b: E4B's log-chance of its real next piece
    s = lp[bi, si][:, SH].log_softmax(-1)
    t = t_sh.log_softmax(-1)
    kl_spot = F.kl_div(s, t, log_target=True, reduction="none").sum(-1)
    use = torch.ones_like(kl_spot, dtype=torch.bool)
    if TEACH == "sel" and need_orig:
        nb = b_ids[bi, (si + 1).clamp(max=b_ids.shape[1] - 1)]
        p_lfm = lp0[bi, si].gather(-1, nb[:, None])[:, 0]
        use = same & (p_e4b > p_lfm)
    kd = (kl_spot * use).sum() / use.sum().clamp(min=1)
    frac = use.float().mean().item()
    if train_mode:
        return ce, kd, leash, frac
    # alignment check: at the paired spots, how often each one's top shared piece is the real next LFM3 piece
    ok = si + 1 < b_m.sum(1)[bi]
    nxt = b_ids[bi, (si + 1).clamp(max=b_ids.shape[1] - 1)]
    return ce, kd, [float(((SH[s.argmax(-1)] == nxt) & ok).sum()), float(((SH[t.argmax(-1)] == nxt) & ok).sum()),
                    float(ok.sum())]


@torch.no_grad()
def exam(tag):
    ce, kd, n, hit = 0.0, 0.0, 0, [0.0, 0.0, 0.0]
    for k in range(0, len(held), 8):
        c, d, h = losses(held[k:k + 8], False)
        ce += c.item(); kd += d.item(); n += 1
        hit = [x + y for x, y in zip(hit, h)]
    drift = 0.0
    if graft is not None:
        for k in range(0, len(held), 8):
            b_ids, b_m = pad([w[0] for w in held[k:k + 8]], tl.pad_token_id)
            lg = student(b_ids).log_softmax(-1)
            hook_off[0] = True
            l0 = lfm.model(input_ids=b_ids).logits.float().log_softmax(-1)
            hook_off[0] = False
            kl = (l0.exp() * (l0 - lg)).sum(-1)
            drift += ((kl * b_m).sum() / b_m.sum()).item()
        drift /= n
    say(f"[{tag}] held-out: LFM3 loss {ce / n:.4f}, distance to E4B {kd / n:.4f}, drift from LFM3 {drift:.4f}; top shared guess = real next piece: LFM3 {hit[0] / hit[2]:.1%}, E4B {hit[1] / hit[2]:.1%}")
    return {"ce": ce / n, "kd": kd / n, "drift": drift, "top_lfm": hit[0] / hit[2], "top_e4b": hit[1] / hit[2]}


res = {"arm": arm, "rooms": ROOMS, "site": SITE, "steps": STEPS, "seed": SEED, "curve": []}
res["start"] = exam("start")
if graft is not None and arm != "host":
    params = [p for p in graft.parameters() if p.requires_grad]
    say(f"trainable numbers: {sum(p.numel() for p in params) / 1e6:.1f}M")
    optim = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    order = list(range(len(train)))
    random.Random(SEED).shuffle(order)
    t0 = time.time()
    for step in range(STEPS):
        batch = [train[order[(step * BS + k) % len(order)]] for k in range(BS)]
        optim.zero_grad(set_to_none=True)
        part = BS // ACC
        for q in range(ACC):
            ce, kd, leash, frac = losses(batch[q * part:(q + 1) * part], True)
            ((ce + KD * kd + LEASH * leash) / ACC).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step(); sched.step()
        if step % 50 == 0 or step == STEPS - 1:
            say(f"  step {step:5}: loss {ce.item():.4f} kd {kd.item():.4f} (on {frac:.0%} of spots) leash {leash.item():.4f}  B size {graft.B.weight.norm():.3f}  "
                f"{time.time() - t0:.0f}s")
            res["curve"].append({"step": step, "ce": ce.item(), "kd": kd.item(), "leash": leash.item(), "taught": frac})
        if (step + 1) % 250 == 0 and step + 1 < STEPS:
            res[f"step{step + 1}"] = exam(f"step {step + 1}")
    torch.save({k: v for k, v in graft.state_dict().items() if k.startswith(("A.", "B."))}, f"{out}/translators.pt")
    res["end"] = exam("end")


# ---------------------------------------------------------------- HellaSwag through the grafted model
@torch.no_grad()
def hellaswag(n):
    hl = open(f"{BENCH}/hellaswag_val_full.txt").read().split("\n")
    right = []
    for k in range(n):
        ctx, lab, ends = hl[6 * k], int(hl[6 * k + 1]), hl[6 * k + 2:6 * k + 6]
        c = [tl.bos_token_id] + tl(ctx, add_special_tokens=False).input_ids
        es = [tl(" " + e, add_special_tokens=False).input_ids for e in ends]
        ids, _ = pad([c + e for e in es], tl.pad_token_id)
        lg = student(ids).log_softmax(-1)
        sc = [lg[j, len(c) - 1:len(c) - 1 + len(e)].gather(-1, torch.tensor(e, device=DEV)[:, None]).mean().item()
              for j, e in enumerate(es)]
        right.append(max(range(4), key=lambda j: sc[j]) == lab)
    return right


if NHS:
    right = hellaswag(NHS)
    gap = json.load(open(f"{DATA}/hs-gap.json"))["gap"] if os.path.exists(f"{DATA}/hs-gap.json") else []
    res["hs"] = sum(right) / len(right)
    res["hs_right"] = right
    gtxt = ""
    if gap:
        res["hs_gap"] = sum(right[k] for k in gap if k < len(right)) / len([k for k in gap if k < len(right)])
        gtxt = f", on the {len(gap)} gap questions {100 * res['hs_gap']:.1f}%"
    say(f"HellaSwag {NHS}: {100 * res['hs']:.1f}%{gtxt}")
json.dump(res, open(f"{out}/result.json", "w"))
say("GRAFT_DONE")
