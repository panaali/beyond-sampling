"""Float64 numpy forward passes of the T432 candidates on truncated Taylor polynomials in the line parameter.

A TP holds, per row, the coefficients of a polynomial in u = t - tc (shape (n, ..., D+1)). Linear maps, sums and products
are exact as long as the true degree stays <= D (products assert that nothing is truncated). Piecewise-linear ops (relu,
clamp, max) take the branch valid just right of u = 0 and record, per row, the smallest positive root at which that
branch changes (`kink`). With D = 0 the same code is a plain forward pass; smooth ops (sigmoid, tanh, softmax) are only
allowed at D = 0. Every op adds its FLOPs (multiply-add = 2) to `flops`, so costs are counted, not estimated.
"""
import numpy as np


class Ctx:
    def __init__(self, n, D):
        self.n, self.D = n, D
        self.kink = np.full(n, np.inf)
        self.flops = 0.0


def poly_eval(c, u):
    """c (..., D+1) ascending; u broadcastable to c[..., 0]."""
    out = np.zeros(np.broadcast_shapes(c[..., 0].shape, np.shape(u)))
    for k in range(c.shape[-1] - 1, -1, -1):
        out = out * u + c[..., k]
    return out


def smallest_pos_root(c, umin=0.0):
    """c (K, D+1): the smallest real root > umin of each polynomial (inf if none). Batched companion eigenvalues,
    grouped by actual degree; roots polished by two Newton steps."""
    K, D1 = c.shape
    out = np.full(K, np.inf)
    if K == 0:
        return out
    if D1 == 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -c[:, 0] / c[:, 1]
        return np.where((r > umin) & np.isfinite(r), r, np.inf)
    scale = np.abs(c).max(1)
    nz = np.abs(c) > 1e-13 * scale[:, None]
    deg = np.where(nz.any(1), D1 - 1 - np.argmax(nz[:, ::-1], 1), 0)
    for d in range(1, D1):
        idx = np.nonzero(deg == d)[0]
        if not len(idx):
            continue
        cc = c[idx, :d + 1] / c[idx, d:d + 1]
        if d == 1:
            roots = -cc[:, :1]
        else:
            M = np.zeros((len(idx), d, d))
            M[:, 1:, :-1] = np.eye(d - 1)
            M[:, :, -1] = -cc[:, :d]
            ev = np.linalg.eigvals(M)
            roots = np.where(np.abs(ev.imag) <= 1e-7 * (1 + np.abs(ev.real)), ev.real, np.nan)
            dc = cc[:, 1:] * np.arange(1, d + 1)
            for _ in range(2):
                f = poly_eval(cc[:, None, :], roots); fp = poly_eval(dc[:, None, :], roots)
                with np.errstate(divide="ignore", invalid="ignore"):
                    step = np.where(np.abs(fp) > 0, f / fp, 0.0)
                roots = roots - np.where(np.isfinite(step), step, 0.0)
        roots = np.where(roots > umin, roots, np.inf)
        out[idx] = np.nanmin(np.where(np.isnan(roots), np.inf, roots), 1)
    return out


class TP:
    def __init__(self, c, ctx):
        self.c, self.ctx = c, ctx

    @property
    def shape(self):
        return self.c.shape[:-1]

    def _new(self, c):
        return TP(c, self.ctx)

    # --- linear ops (exact) ---
    def lin(self, W, b=None):
        """x @ W.T (+ b) along the last data axis."""
        c = np.einsum("...id,oi->...od", self.c, W)
        self.ctx.flops += 2 * np.prod(c.shape[:-1]) * W.shape[1] * (self.ctx.D + 1)
        if b is not None:
            c = c.copy(); c[..., 0] += b
            self.ctx.flops += np.prod(c.shape[:-1])
        return self._new(c)

    def __add__(self, o):
        assert isinstance(o, TP)
        self.ctx.flops += self.c.size
        return self._new(self.c + o.c)

    def addc(self, b):
        """add a constant (to the u^0 coefficient only)."""
        c = self.c.copy(); c[..., 0] += b
        self.ctx.flops += c[..., 0].size
        return self._new(c)

    def scale(self, a):
        """elementwise product with a constant array a (broadcast over rows)."""
        self.ctx.flops += self.c.size
        return self._new(self.c * np.asarray(a)[..., None])

    def mul(self, o):
        """elementwise product of two TPs (truncation is asserted to lose nothing beyond 1e-12 relative)."""
        D = self.ctx.D
        a, b = np.broadcast_arrays(self.c, o.c)
        out = np.zeros(a.shape)
        for i in range(D + 1):
            out[..., i:] += a[..., i:i + 1] * b[..., :D + 1 - i]
        if D > 0:
            lost = sum(np.abs(a[..., i] * b[..., j]).max(initial=0) for i in range(1, D + 1) for j in range(D + 1 - i, D + 1))
            assert lost <= 1e-12 * (1 + np.abs(out).max(initial=0)), "degree overflow: raise D"
        self.ctx.flops += a[..., 0].size * (D + 1) * (D + 2)
        return self._new(out)

    def matmul(self, o):
        """batched (..., m, k) @ (..., k, p) on TPs."""
        D = self.ctx.D
        m, k = self.c.shape[-3], self.c.shape[-2]; p = o.c.shape[-2]
        out = np.zeros(np.broadcast_shapes(self.c.shape[:-3], o.c.shape[:-3]) + (m, p, D + 1))
        for i in range(D + 1):
            for j in range(D + 1 - i):
                out[..., i + j] += self.c[..., i] @ o.c[..., j]
        if D > 0:
            for i in range(1, D + 1):
                for j in range(D + 1 - i, D + 1):
                    v = np.abs(self.c[..., i] @ o.c[..., j]).max(initial=0)
                    assert v <= 1e-12 * (1 + np.abs(out).max(initial=0)), "degree overflow: raise D"
        self.ctx.flops += 2 * out[..., 0].size * k * (D + 1) * (D + 2) / 2
        return self._new(out)

    def T(self):
        return self._new(np.swapaxes(self.c, -2, -3))

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        return self._new(self.c[idx + (Ellipsis,)] if Ellipsis not in idx else self.c[idx])

    # --- piecewise-linear ops (branch at u = 0+, kink recorded) ---
    def _sign0(self, c):
        """sign of the polynomial just right of u = 0 (first nonzero coefficient)."""
        s = np.sign(c[..., 0])
        for k in range(1, c.shape[-1]):
            s = np.where(s == 0, np.sign(c[..., k]), s)
        return s

    def _kinks(self, c):
        rows = c.shape[0]
        flat = c.reshape(rows, -1, c.shape[-1])
        r = smallest_pos_root(flat.reshape(-1, c.shape[-1])).reshape(rows, -1).min(1) if flat.shape[1] else np.full(rows, np.inf)
        self.ctx.kink = np.minimum(self.ctx.kink, r)
        D = self.ctx.D
        self.ctx.flops += flat[..., 0].size * (2 if D <= 1 else 10 * D ** 3)

    def relu(self):
        if self.ctx.D > 0:
            self._kinks(self.c)
        on = self._sign0(self.c) > 0
        self.ctx.flops += self.c[..., 0].size
        return self._new(np.where(on[..., None], self.c, 0.0))

    def clamp(self, lo, hi):
        c = self.c
        if self.ctx.D > 0:
            a = c.copy(); a[..., 0] -= lo; self._kinks(a)
            b = c.copy(); b[..., 0] -= hi; self._kinks(b)
        s_lo = self._sign0(c - np.r_[lo, np.zeros(c.shape[-1] - 1)])
        s_hi = self._sign0(c - np.r_[hi, np.zeros(c.shape[-1] - 1)])
        out = np.where((s_lo <= 0)[..., None], np.r_[lo, np.zeros(c.shape[-1] - 1)], c)
        out = np.where((s_hi >= 0)[..., None], np.r_[hi, np.zeros(c.shape[-1] - 1)], out)
        self.ctx.flops += 2 * c[..., 0].size
        return self._new(out)

    def maxpool(self, axis):
        """max over a data axis (not the row axis 0)."""
        c = np.moveaxis(self.c, axis, -2)                 # (..., K, D+1)
        K = c.shape[-2]
        # argmax just right of 0: lexicographic on coefficients
        best = np.zeros(c.shape[:-2], int)
        for k in range(1, K):
            diff = c[..., k, :] - np.take_along_axis(c, best[..., None, None], -2)[..., 0, :]
            best = np.where(self._sign0(diff) > 0, k, best)
        top = np.take_along_axis(c, best[..., None, None], -2)[..., 0, :]
        if self.ctx.D > 0:
            self._kinks((c - top[..., None, :]).reshape(c.shape[0], -1, c.shape[-1]))
        self.ctx.flops += c[..., 0].size * 2
        return self._new(top)

    # --- smooth ops (D = 0 only) ---
    def _smooth(self, f, k=20):
        assert self.ctx.D == 0, "smooth op in a polynomial walk: use the numerical walk"
        self.ctx.flops += k * self.c.size
        return self._new(f(self.c))

    def sigmoid(self):
        return self._smooth(lambda c: 1 / (1 + np.exp(-c)))

    def tanh(self):
        return self._smooth(np.tanh)

    def softmax(self, axis):
        def f(c):
            z = c[..., 0]; z = z - z.max(axis, keepdims=True); e = np.exp(z)
            return (e / e.sum(axis, keepdims=True))[..., None]
        return self._smooth(f, 25)


def line_input(y, d, tc, D, ctx):
    """x(u) = y + (tc + u) d as a TP of degree D. y, d (n, T); tc (n,)."""
    c = np.zeros(y.shape + (D + 1,))
    c[..., 0] = y + tc[:, None] * d
    if D >= 1:
        c[..., 1] = d
    return TP(c, ctx)


# ---------------- the candidates (weights: dict of float64 arrays from load.py) ----------------
def rnn(w, x):
    H = w["Wh"].shape[0]; n = x.c.shape[0]; T = x.c.shape[1]
    h = TP(np.zeros((n, H, x.c.shape[-1])), x.ctx)
    for s in range(T):
        z = h.lin(w["Wh"], w.get("bh")) + x[:, s:s + 1].lin(w["Wi"], w.get("bi"))
        h = z.relu()
    return h.lin(w["Wo"], w.get("bo"))


def rnn2r(w, x):
    H = w["Wh1"].shape[0]; n = x.c.shape[0]; T = x.c.shape[1]
    z0 = np.zeros((n, H, x.c.shape[-1]))
    h1, h2 = TP(z0, x.ctx), TP(z0.copy(), x.ctx)
    for s in range(T):
        h1 = (h1.lin(w["Wh1"]) + x[:, s:s + 1].lin(w["Wi1"])).relu()
        h2 = (h2.lin(w["Wh2"]) + h1.lin(w["Wi2"])).relu()
    return (h1 + h2).lin(w["Wo"])


def gru(w, x, hard):
    H = w["Wh"].shape[0] // 3; n = x.c.shape[0]; T = x.c.shape[1]
    h = TP(np.zeros((n, H, x.c.shape[-1])), x.ctx)
    sg = (lambda z: z.scale(1 / 6).addc(0.5).clamp(0.0, 1.0)) if hard else (lambda z: z.sigmoid())
    tn = (lambda z: z.clamp(-1.0, 1.0)) if hard else (lambda z: z.tanh())
    for s in range(T):
        gx = x[:, s:s + 1].lin(w["Wx"], w["bx"]); gh = h.lin(w["Wh"], w["bh"])
        r = sg(gx[:, :H] + gh[:, :H]); z = sg(gx[:, H:2 * H] + gh[:, H:2 * H])
        nn_ = tn(gx[:, 2 * H:] + r.mul(gh[:, 2 * H:]))
        h = z.scale(-1.0).addc(1.0).mul(nn_) + z.mul(h)
    return h.lin(w["Wo"], w["bo"])


def ssm(w, x):
    n = x.c.shape[0]; T = x.c.shape[1]; N = w["a1"].shape[0]
    s = TP(np.zeros((n, N, x.c.shape[-1])), x.ctx); s2 = TP(np.zeros_like(s.c), x.ctx)
    for t in range(T):
        s = s.scale(w["a1"]) + x[:, t:t + 1].lin(w["B1"])
        v = s + s.lin(w["m1"]).relu().lin(w["m2"])
        s2 = s2.scale(w["a2"]) + v.lin(w["B2"])
    return s2.lin(w["o1"]).relu().lin(w["o2"])


def conv(w, x):
    n, T = x.c.shape[0], x.c.shape[1]
    f = TP(x.c[:, None], x.ctx)                       # (n, C=1, T, D+1)
    for i, d in enumerate((1, 2, 4, 8)):
        W = w[f"c{i}"]                                # (Cout, Cin, 3)
        Cin = W.shape[1]
        pad = np.zeros((n, Cin, d, f.c.shape[-1]))
        fp = np.concatenate([pad, f.c, pad], 2)
        out = 0
        for j in range(3):
            out = out + np.einsum("nitd,oi->notd", fp[:, :, j * d:j * d + T], W[:, :, j])
        x.ctx.flops += 2 * n * W.shape[0] * Cin * 3 * T * (x.ctx.D + 1)
        f = TP(out, x.ctx).relu()
    g = f.maxpool(2)                                  # (n, C, D+1)
    ft = np.swapaxes(f.c, 1, 2)                       # (n, T, C, D+1)
    z = TP(np.concatenate([ft, np.broadcast_to(g.c[:, None], ft.shape)], 2), x.ctx)
    return z.lin(w["h1"]).relu().lin(w["h2"])[:, :, 0]


def tf(w, x, relu_attn):
    n, T = x.c.shape[0], x.c.shape[1]; nh = 2; d = w["e2"].shape[0]; dh = d // nh
    h = TP(x.c[:, :, None], x.ctx).lin(w["e1"], w["e1b"]).relu().lin(w["e2"], w["e2b"])   # (n, T, d)
    for l in range(w["L"]):
        q = h.lin(w[f"q{l}"]); k = h.lin(w[f"k{l}"]); v = h.lin(w[f"v{l}"])
        heads = []
        for hd in range(nh):
            sl = slice(hd * dh, (hd + 1) * dh)
            qh, kh, vh = q[:, :, sl], k[:, :, sl], v[:, :, sl]
            sc = qh.matmul(kh.T()).scale(1 / dh ** 0.5)          # (n, T, T)
            A = sc.relu().scale(1 / T) if relu_attn else sc.softmax(2)
            heads.append(A.matmul(vh).c)
        att = TP(np.concatenate(heads, 2), x.ctx)
        h = h + att.lin(w[f"o{l}"])
        h = h + h.lin(w[f"f1{l}"], w[f"f1b{l}"]).relu().lin(w[f"f2{l}"], w[f"f2b{l}"])
    return h.lin(w["out"], w["outb"])[:, :, 0]


FORWARD = {"rnn": rnn, "rnn2r": rnn2r, "hgru": lambda w, x: gru(w, x, True), "sgru": lambda w, x: gru(w, x, False),
           "ssm": ssm, "conv": conv, "tfrelu": lambda w, x: tf(w, x, True), "tfsoft": lambda w, x: tf(w, x, False)}
# walk mode: exact degree (PL = 1, ReLU-attention 2 layers = 9) or "num" (smooth / products of unbounded degree)
MODE = {"rnn": 1, "rnn2r": 1, "ssm": 1, "conv": 1, "tfrelu": 9, "hgru": "num", "sgru": "num", "tfsoft": "num"}
HOMOG = {"rnn": True, "rnn2r": True, "ssm": True, "conv": True, "tfrelu": False, "hgru": False, "sgru": False, "tfsoft": False}


def forward(kind, w, X, D=0):
    """plain float64 forward: X (n, T) -> logits (n, T), and FLOPs per row."""
    ctx = Ctx(len(X), D)
    L = FORWARD[kind](w, line_input(X, np.zeros_like(X), np.zeros(len(X)), D, ctx))
    return L.c[..., 0], ctx.flops / len(X)
