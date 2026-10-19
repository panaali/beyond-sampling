"""T432 candidate architectures (torch, for training). All map x (B, 10) -> logits (B, 10).

The numpy walk versions live in netnp.py; `export(model)` returns the float64 weight dict they read.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as Fn

T = 10


def hsig(x):
    return th.clamp(x / 6 + 0.5, 0.0, 1.0)


class RNN(nn.Module):
    """alg_zoo DistRNN layout (th.nn.RNN relu + linear), optional biases. h_0 = 0, logits = W_out h_T (+ b)."""
    def __init__(self, H=32, bias=False):
        super().__init__()
        self.rnn = nn.RNN(1, H, nonlinearity="relu", bias=bias, batch_first=True)
        self.linear = nn.Linear(H, T, bias=bias)

    def forward(self, x):
        h, _ = self.rnn(x[..., None])
        return self.linear(h[:, -1])


class RNN2R(nn.Module):
    """Two stacked bias-free ReLU RNN layers, residual readout W_out (h1_T + h2_T)."""
    def __init__(self, H=32):
        super().__init__()
        self.r1 = nn.RNN(1, H, nonlinearity="relu", bias=False, batch_first=True)
        self.r2 = nn.RNN(H, H, nonlinearity="relu", bias=False, batch_first=True)
        self.linear = nn.Linear(H, T, bias=False)

    def forward(self, x):
        h1, _ = self.r1(x[..., None])
        h2, _ = self.r2(h1)
        return self.linear(h1[:, -1] + h2[:, -1])


class GRU(nn.Module):
    """GRU (torch gate convention) with hard (hard-sigmoid / hard-tanh) or smooth (sigmoid / tanh) gates."""
    def __init__(self, H=32, hard=True):
        super().__init__()
        self.H, self.hard = H, hard
        self.Wx = nn.Linear(1, 3 * H)            # r, z, n (with bias)
        self.Wh = nn.Linear(H, 3 * H)            # r, z, n (with bias)
        self.linear = nn.Linear(H, T)

    def forward(self, x):
        B, H = x.shape[0], self.H
        sg = hsig if self.hard else th.sigmoid
        th_ = (lambda u: th.clamp(u, -1.0, 1.0)) if self.hard else th.tanh
        h = x.new_zeros(B, H)
        gx = self.Wx(x[..., None])               # (B, T, 3H)
        for s in range(T):
            gh = self.Wh(h)
            r = sg(gx[:, s, :H] + gh[:, :H])
            z = sg(gx[:, s, H:2 * H] + gh[:, H:2 * H])
            n = th_(gx[:, s, 2 * H:] + r * gh[:, 2 * H:])
            h = (1 - z) * n + z * h
        return self.linear(h)


class SSM(nn.Module):
    """Bias-free 2-block diagonal linear SSM (LRU-style, non-selective) with ReLU MLPs."""
    def __init__(self, N=32, M=64):
        super().__init__()
        def nu_init():
            a = th.rand(N) * 0.45 + 0.5          # decay in [0.5, 0.95]
            return nn.Parameter(th.log(-th.log(a)))
        self.nu1, self.nu2 = nu_init(), nu_init()
        self.B1 = nn.Linear(1, N, bias=False)
        self.m1 = nn.Linear(N, M, bias=False)
        self.m2 = nn.Linear(M, N, bias=False)
        self.B2 = nn.Linear(N, N, bias=False)
        self.o1 = nn.Linear(N, M, bias=False)
        self.o2 = nn.Linear(M, T, bias=False)

    def forward(self, x):
        a1, a2 = th.exp(-th.exp(self.nu1)), th.exp(-th.exp(self.nu2))
        u = self.B1(x[..., None])                # (B, T, N)
        s = th.zeros_like(u[:, 0]); s2 = th.zeros_like(s)
        for t in range(T):
            s = a1 * s + u[:, t]
            v = s + self.m2(th.relu(self.m1(s)))
            s2 = a2 * s2 + self.B2(v)
        return self.o2(th.relu(self.o1(s2)))


class Conv(nn.Module):
    """Bias-free dilated 1-D conv net, global max-pool, per-position ReLU head."""
    def __init__(self, C=32, K=32):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(1 if i == 0 else C, C, 3, dilation=d, padding=d, bias=False)
                                    for i, d in enumerate((1, 2, 4, 8))])
        self.h1 = nn.Linear(2 * C, K, bias=False)
        self.h2 = nn.Linear(K, 1, bias=False)

    def forward(self, x):
        f = x[:, None, :]
        for c in self.convs:
            f = th.relu(c(f))                    # (B, C, T)
        g = f.max(-1).values                     # (B, C)
        z = th.cat([f.transpose(1, 2), g[:, None, :].expand(-1, T, -1)], -1)
        return self.h2(th.relu(self.h1(z)))[..., 0]


class TF(nn.Module):
    """2-layer transformer, no positional embedding, no norm; ReLU or softmax attention."""
    def __init__(self, d=32, heads=2, L=2, relu_attn=True, E=32, M=64):
        super().__init__()
        self.d, self.nh, self.relu_attn = d, heads, relu_attn
        self.e1 = nn.Linear(1, E); self.e2 = nn.Linear(E, d)
        self.q = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.k = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.v = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.o = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.f1 = nn.ModuleList([nn.Linear(d, M) for _ in range(L)])
        self.f2 = nn.ModuleList([nn.Linear(M, d) for _ in range(L)])
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        B = x.shape[0]; nh = self.nh; dh = self.d // nh
        h = self.e2(th.relu(self.e1(x[..., None])))          # (B, T, d)
        for l in range(len(self.q)):
            q = self.q[l](h).view(B, T, nh, dh).transpose(1, 2)
            k = self.k[l](h).view(B, T, nh, dh).transpose(1, 2)
            v = self.v[l](h).view(B, T, nh, dh).transpose(1, 2)
            sc = q @ k.transpose(-1, -2) / dh ** 0.5            # (B, nh, T, T)
            A = th.relu(sc) / T if self.relu_attn else th.softmax(sc, -1)
            h = h + self.o[l]((A @ v).transpose(1, 2).reshape(B, T, self.d))
            h = h + self.f2[l](th.relu(self.f1[l](h)))
        return self.out(h)[..., 0]


def build(tag):
    return {
        "rnn16": lambda: RNN(16, bias=False),     # loop check only
        "rnnb32": lambda: RNN(32, bias=True),
        "rnn2r32": lambda: RNN2R(32),
        "hgru32": lambda: GRU(32, hard=True),
        "sgru32": lambda: GRU(32, hard=False),
        "ssm": lambda: SSM(),
        "ssmB": lambda: SSM(),                    # A1 fallback retrain of ssm at lr_base 0.012
        "conv": lambda: Conv(),
        "tfrelu": lambda: TF(relu_attn=True),
        "tfsoft": lambda: TF(relu_attn=False),
    }[tag]()


WIDTH = {"rnn16": 16, "rnnb32": 32, "rnn2r32": 32, "hgru32": 32, "sgru32": 32, "ssm": 32, "ssmB": 32, "conv": 32,
         "tfrelu": 32, "tfsoft": 32}


def nparams(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    x = th.randn(4, T)
    for t in WIDTH:
        m = build(t)
        print(t, nparams(m), tuple(m(x).shape))
