"""The gated brake (Immy, 2026-10-28): "when enough of those flags hit ... route the be-brief neurons to activate more".

A detector reads the stream after one layer (overthink_probe.py's direction, built on words at the same point of
their thinking). Its reading is smoothed over the last few words (enough flags, not one), and while the smoothed
reading is past the set point between "working" and "already solved", the "Be brief." change (brief_patient.py, its
average size per word at that layer) is added to the stream in front of a later layer. Only while thinking, and only
on new words (never on the prompt). mode=always adds it on every thinking word (the plain agonist, no gate).
spread=1 adds the change across every layer instead: each layer gets its own step of it (the change at its input
minus the change at the layer before), so the whole stream moves the way the instruction moves it, not one spot.
close=B pushes on closing the thinking itself instead: while the gate is open, the end-of-thinking word's score gets
+B (log-odds), so the patient closes at its next natural stop rather than re-checking (test 2: the "brief" signal is
too weak in LFM3 to lean on). Baked, this is one neuron whose output writes along that word's row of the unembedding.
clock=<clock_dirs.pt> (clock_probe.py) replaces the detector with the patient's own clock: the straight-line reading
of "thinking, past word W" at the input of layer det's MLP, one word at a time, no smoothing and no counter, exactly
what one added neuron in that MLP could compute. Its own set point decides; warm is ignored.
warm=N keeps the gate shut for the first N thinking words (test 3: at word 1 the detector reads the question, not
progress, and is open on every problem, so a strong close= just switches thinking off).
Run-time hooks first, to test the idea; the baked version (a neuron in the weights) comes after it earns it.

    from brake import Brake
    b = Brake(p, overthink_dirs, brief_dirs, det=8, push=12, alpha=1.0, mode="gate", ema=16)
    model.generate(..., logits_processor=[b])      # b also watches for the end of thinking
"""
import torch
from transformers import LogitsProcessor


class Brake(LogitsProcessor):
    def __init__(self, p, od, bd, det=8, push=12, alpha=1.0, mode="gate", ema=16, margin=0.0, spread=False, close=0.0, warm=0, clock=None):
        lay = p.model.model.layers
        dev = next(p.model.parameters()).device
        self.w = od[det]["dir_bin"].to(dev)
        self.set = od[det]["mid_bin"] + margin * od[det]["sd_bin"]
        self.vec = (alpha * bd[push]["mean"]).to(dev)
        self.mode, self.k, self.end, self.close = mode, 2 / (ema + 1), p.end, close
        self.warm = warm
        self.ema = self.open = self.think = None
        self.stats = None
        self.clock = clock
        if clock is None:
            lay[det].register_forward_hook(self._read)
        else:
            c = clock[det]
            self.cw, self.cb, self.cthr = c["w"].to(dev), c["b"], c["thr"]
            lay[det].feed_forward.register_forward_pre_hook(self._read_clock)
        if not spread:
            lay[push].register_forward_pre_hook(lambda m, a, kw: self._push(a, kw, self.vec), with_kwargs=True)
        else:                                                 # read happens after layer det; earlier layers push
            for i in range(1, len(lay)):                      # with the gate as it stood after the previous word
                step = (alpha * (bd[i]["mean"] - bd[i - 1]["mean"])).to(dev)
                lay[i].register_forward_pre_hook(lambda m, a, kw, v=step: self._push(a, kw, v), with_kwargs=True)

    def _read(self, m, a, o):
        h = o[0] if isinstance(o, tuple) else o
        if h.shape[1] > 1:                                    # the prompt: start fresh
            B = h.shape[0]
            self.ema = torch.full((B,), float("nan"), device=h.device)
            self.open = torch.zeros(B, dtype=torch.bool, device=h.device)
            self.think = torch.ones(B, dtype=torch.bool, device=h.device)
            self.stats = torch.zeros(B, 2, device=h.device)  # thinking words, of which braked
            return
        s = h[:, 0].float() @ self.w
        self.ema = torch.where(torch.isnan(self.ema), s, self.ema + self.k * (s - self.ema))
        self.open = self.ema > self.set if self.mode == "gate" else torch.ones_like(self.open)

    def _read_clock(self, m, a):
        x = a[0]
        if x.shape[1] > 1:
            return self._read(m, a, x)                        # the prompt: start fresh
        self.open = (x[:, 0].float() @ self.cw + self.cb) > self.cthr

    def _push(self, a, kw, vec):
        h = a[0] if a else kw["hidden_states"]
        if h.shape[1] > 1 or self.mode == "off" or self.open is None:
            return
        on = (self.open & self.think).to(h.dtype)[:, None, None]
        h = h + on * vec.to(h.dtype)
        if a:
            return (h,) + tuple(a[1:]), kw
        kw["hidden_states"] = h
        return a, kw

    def __call__(self, input_ids, scores):                   # sees each new word: thinking ends at the end word
        if self.think is not None:
            self.stats[:, 0] += self.think.float(); self.stats[:, 1] += (self.open & self.think).float()
            self.think &= input_ids[:, -1] != self.end
            if self.close:
                on = self.open & self.think & (self.stats[:, 0] > (0 if self.clock is not None else self.warm))
                scores[:, self.end] += self.close * on.to(scores.dtype)
        return scores
