"""Tiny next-token models (T433): a ReLU RNN and a no-layer-norm transformer with attention variants.

Attention variants (`attn`):
  softmax  reference
  hard     average-hard attention: uniform over the argmax set of the scores (ties within 1e-6);
           trained straight-through (forward hard, backward softmax); eval is pure hard
  relu     relu(score) / t   (t = number of visible positions)
  linear   score / t
  pos      softmax(A[t, :t]) with a learned matrix A: input-independent mixing
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLURNN(nn.Module):
    def __init__(self, V_in, V_out, H=64, **_):
        super().__init__()
        self.cfg = dict(arch="rnn", V_in=V_in, V_out=V_out, H=H)
        self.U = nn.Embedding(V_in, H)
        self.W = nn.Linear(H, H)
        self.out = nn.Linear(H, V_out)
        nn.init.normal_(self.W.weight, std=0.5 / math.sqrt(H))

    def forward(self, x, return_hidden=False):
        B, T = x.shape
        u = self.U(x)
        h = torch.zeros(B, self.W.in_features, device=x.device)
        hs = []
        for t in range(T):
            h = F.relu(self.W(h) + u[:, t])
            hs.append(h)
        hs = torch.stack(hs, 1)
        return (self.out(hs), hs) if return_hidden else self.out(hs)

    def flops(self, T):
        H, V = self.cfg["H"], self.cfg["V_out"]
        return T * (2 * H * H + 2 * H + 2 * H * V)


class Block(nn.Module):
    def __init__(self, d, mlp, attn, Lmax):
        super().__init__()
        self.attn = attn
        if attn == "pos":
            self.A = nn.Parameter(torch.zeros(Lmax, Lmax))
        else:
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d)
        self.f1 = nn.Linear(d, mlp)
        self.f2 = nn.Linear(mlp, d)
        self.hard_eval = True

    def weights(self, h):
        B, T, d = h.shape
        mask = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
        if self.attn == "pos":
            s = self.A[:T, :T].masked_fill(~mask, float("-inf"))
            return torch.softmax(s, -1).expand(B, T, T)
        s = self.q(h) @ self.k(h).transpose(1, 2) / math.sqrt(d)
        cnt = torch.arange(1, T + 1, device=h.device, dtype=h.dtype)[:, None]
        if self.attn == "softmax":
            return torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
        if self.attn == "relu":
            return F.relu(s).masked_fill(~mask, 0.0) / cnt
        if self.attn == "linear":
            return s.masked_fill(~mask, 0.0) / cnt
        if self.attn == "hard":
            sm = s.masked_fill(~mask, float("-inf"))
            mx = sm.max(-1, keepdim=True).values
            hard = (sm >= mx - 1e-6).to(h.dtype)
            hard = hard / hard.sum(-1, keepdim=True)
            if self.training:
                soft = torch.softmax(sm, -1)
                return hard + soft - soft.detach()
            return hard
        raise ValueError(self.attn)

    def forward(self, h):
        a = self.weights(h)
        h = h + self.o(a @ self.v(h))
        return h + self.f2(F.relu(self.f1(h)))


class TinyTransformer(nn.Module):
    def __init__(self, V_in, V_out, Lmax, d=32, layers=2, mlp=64, attn="softmax", **_):
        super().__init__()
        self.cfg = dict(arch=attn, V_in=V_in, V_out=V_out, Lmax=Lmax, d=d, layers=layers, mlp=mlp, attn=attn)
        self.tok = nn.Embedding(V_in, d)
        self.pos = nn.Embedding(Lmax, d)
        nn.init.normal_(self.tok.weight, std=0.5)
        nn.init.normal_(self.pos.weight, std=0.5)
        self.blocks = nn.ModuleList(Block(d, mlp, attn, Lmax) for _ in range(layers))
        self.out = nn.Linear(d, V_out)

    def forward(self, x, return_hidden=False):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        hs = [h]
        for b in self.blocks:
            h = b(h)
            hs.append(h)
        return (self.out(h), hs) if return_hidden else self.out(h)

    def flops(self, T):
        c = self.cfg
        d, m, V = c["d"], c["mlp"], c["V_out"]
        proj = 2 if c["attn"] == "pos" else 4  # pos has no q, k (A is a lookup) but keeps v, o
        mix = 2 * d * T * (T + 1) // 2 * (1 if c["attn"] == "pos" else 2)  # (scores +) weighted sum
        per_layer = T * proj * 2 * d * d + mix + T * 4 * d * m
        return c["layers"] * per_layer + T * 2 * d * V


ARCHS = ["rnn", "softmax", "hard", "relu", "linear", "pos"]


def build(arch, lang):
    if arch == "rnn":
        return ReLURNN(lang.V_in, lang.V_out)
    return TinyTransformer(lang.V_in, lang.V_out, lang.L, attn=arch)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
