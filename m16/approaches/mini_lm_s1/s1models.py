"""S1 attention variants (T444). Subclasses the T433 harness Block / TinyTransformer read-only.

Per-layer attention types (`attn` = comma list, one per layer, e.g. "pos,hard"):
  hard      hard argmax attention, uniform over scores within `tol` of the max (tol 1e-6 = the harness
            `hard`; tol > 0 = AHAT tie tolerance). Eval is pure hard. Training mode is set by the trainer:
            'st' (forward hard, backward softmax), 'anneal' (softmax(beta*s)), 'mix' ((1-lam) soft + lam st).
  sparsemax sparsemax(beta*s) in train and eval (piecewise linear in the scores; beta fixed after training)
  keyhard   hard, but scores depend only on the key: s_ij = w . k(h_j) (no query)
  keysparse sparsemax over key-only scores
  pos       harness pos (softmax over a learned position matrix; input independent)
  softmax   harness softmax (reference)
"""
import math, os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mini_lm"))
from models import Block, TinyTransformer  # noqa: E402


def sparsemax(z, dim=-1):
    zs, _ = torch.sort(z, dim=dim, descending=True)
    rng = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = -1
    rng = rng.view(shape)
    zs_f = torch.where(torch.isfinite(zs), zs, torch.zeros_like(zs))
    cs = zs_f.cumsum(dim)
    support = (1 + rng * zs > cs) & torch.isfinite(zs)
    k = support.sum(dim, keepdim=True).clamp(min=1)
    tau = (cs.gather(dim, k - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class S1Block(Block):
    def __init__(self, d, mlp, attn, Lmax, tol=1e-6):
        base = "pos" if attn == "pos" else "softmax"
        super().__init__(d, mlp, base, Lmax)
        self.attn = attn
        self.tol = tol
        self.mode, self.beta, self.lam = "st", 1.0, 1.0
        if attn in ("keyhard", "keysparse"):
            del self.q
            self.w = nn.Parameter(torch.randn(d) / math.sqrt(d))

    def scores(self, h):
        d = h.shape[-1]
        if self.attn.startswith("key"):
            s = (self.k(h) @ self.w)[:, None, :] / math.sqrt(d)
            return s.expand(-1, h.shape[1], -1)
        return self.q(h) @ self.k(h).transpose(1, 2) / math.sqrt(d)

    def weights(self, h):
        if self.attn in ("pos", "softmax"):
            return super().weights(h)
        T = h.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
        sm = self.scores(h).masked_fill(~mask, float("-inf"))
        if self.attn.endswith("sparse") or self.attn == "sparsemax":
            return sparsemax(self.beta * sm)
        mx = sm.max(-1, keepdim=True).values
        hard = (sm >= mx - self.tol).to(h.dtype)
        hard = hard / hard.sum(-1, keepdim=True)
        if not self.training:
            return hard
        soft = torch.softmax(self.beta * sm, -1)
        if self.mode == "anneal":
            return soft
        st = hard + soft - soft.detach()
        if self.mode == "mix":
            return (1 - self.lam) * soft + self.lam * st
        return st


class S1Transformer(TinyTransformer):
    def __init__(self, V_in, V_out, Lmax, d=32, mlp=64, attn="softmax,softmax", tol=1e-6, **_):
        types = attn.split(",")
        super().__init__(V_in, V_out, Lmax, d=d, layers=len(types), mlp=mlp, attn="softmax")
        self.cfg = dict(arch="s1", V_in=V_in, V_out=V_out, Lmax=Lmax, d=d, mlp=mlp, attn=attn, tol=tol)
        self.blocks = nn.ModuleList(S1Block(d, mlp, a, Lmax, tol) for a in types)

    def set_train_mode(self, mode, beta=1.0, lam=1.0):
        for b in self.blocks:
            b.mode, b.beta, b.lam = mode, beta, lam

    def flops(self, T):
        c = self.cfg
        d, m, V = c["d"], c["mlp"], c["V_out"]
        tot = T * 2 * d * V
        for a in c["attn"].split(","):
            proj = 2 if a == "pos" else (3 if a.startswith("key") else 4)
            mix = 2 * d * T * (T + 1) // 2 * (1 if a == "pos" else 2)
            tot += T * proj * 2 * d * d + mix + T * 4 * d * m
        return tot
