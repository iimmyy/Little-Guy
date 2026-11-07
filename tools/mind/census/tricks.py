"""The run-time tricks from the 2026-10-26 hunt, as one switchboard on a loaded E4B (e4b_mac.load()):

    T = Tricks(model)
    T.set(route=set(range(7)), keep=0.2, zero=0.15, swap={22: (A, b)})
    T.set()                                   # all off: the untouched model
    lines = T.fit(windows, route, keep, zero) # background lines for the gate pick (gate_line.py's recipe)
    T.set(route=..., keep=..., zero=..., pick="gate", lines=lines)

  route/keep  in these layers run only the best `keep` share of neurons for each word
  pick        "perfect": by each neuron's real output size (upper bound); "gate": from the gates alone,
              |gelu(gate . x)| x |up row| x |down column|, what a real kernel can know before fetching up/down
  lines       per routed layer, a least-squares straight line (out += x A + b) standing in for the skipped neurons
  zero        zero the weakest `zero` share of each MLP's input, all layers
  swap        replace these layers' whole MLP with a straight line  out = x A + b
"""
import torch
import torch.nn.functional as F


class Tricks:
    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.N = self.layers[0].mlp.intermediate_size
        self.dn = [l.mlp.down_proj.weight.float().norm(dim=0) for l in self.layers]
        self.un = [l.mlp.up_proj.weight.float().norm(dim=1) for l in self.layers]
        self.x = [None] * len(self.layers)
        self.acc = None
        self.set()
        for i, l in enumerate(self.layers):
            l.mlp.register_forward_pre_hook(self._keep_x(i))
            l.mlp.gate_proj.register_forward_pre_hook(self._zero)
            l.mlp.up_proj.register_forward_pre_hook(self._zero)
            l.mlp.down_proj.register_forward_pre_hook(self._route(i))
            l.mlp.register_forward_hook(self._after(i))

    def set(self, route=(), keep=1.0, zero=0.0, swap=None, pick="perfect", lines=None):
        self.route, self.keep, self.zero, self.swap = set(route), keep, zero, swap or {}
        self.pick, self.lines = pick, lines or {}

    def _keep_x(self, i):
        def h(mod, args):
            self.x[i] = args[0]
        return h

    def _zero(self, mod, args):
        if not self.zero:
            return None
        x = args[0]
        thr = x.abs().float().kthvalue(int(self.zero * x.shape[-1]), dim=-1, keepdim=True).values
        return (x * (x.abs() > thr),)

    def _mask(self, i, p):
        if self.pick == "gate":
            g = F.gelu(self.layers[i].mlp.gate_proj(self.x[i]).float(), approximate="tanh")
            size = g.abs() * self.un[i] * self.dn[i]
        else:
            size = p.float().abs() * self.dn[i]
        return torch.zeros_like(p).scatter_(-1, size.topk(int(self.keep * self.N), dim=-1).indices, 1)

    def _route(self, i):
        def h(mod, args):
            if i not in self.route:
                return None
            p = args[0]
            m = self._mask(i, p)
            if self.acc is not None:                                    # fitting: what the skipped neurons add
                r = ((p * (1 - m)).float() @ mod.weight.float().T)[0, 1:].cpu().double()
                x = self.x[i][0, 1:].float()
                x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], 1).cpu().double()
                a = self.acc.setdefault(i, [0, 0])
                a[0] = a[0] + x.T @ x
                a[1] = a[1] + x.T @ r
                return None
            return (p * m,)
        return h

    def _after(self, i):
        def h(mod, args, o):
            if i in self.swap:
                A, b = self.swap[i]
                return (args[0].float() @ A + b).to(o.dtype)
            if i in self.lines and i in self.route and self.acc is None:
                A, b = self.lines[i]
                return o + (args[0].float() @ A + b).to(o.dtype)
            return None
        return h

    def fit(self, windows, route, keep, zero=0.0, pick="gate"):
        """least-squares background line per routed layer, fitted on these token windows with the zeroing on and
        no routing applied while fitting (gate_line.py's recipe)"""
        self.set(route=route, keep=keep, zero=zero, pick=pick)
        self.acc = {}
        dev = self.layers[0].mlp.down_proj.weight.device
        with torch.no_grad():
            for w in windows:
                self.model.model(input_ids=torch.tensor([w], device=dev))
                torch.mps.empty_cache()
        lines = {}
        for i, (xtx, xtr) in self.acc.items():
            D = xtr.shape[1]
            reg = 1e-4 * xtx.diagonal()[:D].mean() * torch.eye(D + 1, dtype=torch.float64)
            reg[D, D] = 0
            Wb = torch.linalg.solve(xtx + reg, xtr).float()
            lines[i] = (Wb[:D].to(dev), Wb[D].to(dev))
        self.acc = None
        return lines
