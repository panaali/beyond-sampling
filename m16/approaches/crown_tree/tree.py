"""T279: certification tree on the sphere with CROWN (backward linear relaxation) or IBP bounds.

Cones are simplicial, given by unit generators R (d x d, rows). The margin sign on a cone = sign on the simplex
conv(R) (no biases => positive homogeneity), so every linear bound is minimised over the d vertices.
"""
import heapq
import itertools
import math
import os
import sys
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/user/lb3/m16/approaches/cone_mass4")
import mass4 as M4  # noqa: E402  (T109 exact simplicial cone mass, d = 4)


def load(path):
    sd = torch.load(path, weights_only=True, map_location="cpu")
    f = lambda k: sd[k].double().numpy()
    return f("rnn.weight_ih_l0")[:, 0], f("rnn.weight_hh_l0"), f("linear.weight")


def forward_argmax(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


def roots(T):
    """2 T (T-1) simplicial cones: truth region (i, j) times the +/- halves of the line R 1. Label = j."""
    out = []
    one = np.ones(T) / math.sqrt(T)
    for i, j in itertools.permutations(range(T), 2):
        gens = [np.eye(T)[i]] + [-np.eye(T)[k] for k in range(T) if k not in (i, j)]
        for s in (+1, -1):
            out.append((np.array(gens + [s * one]), j, (i, j, s)))
    return out


class Flops:
    def __init__(self):
        self.n = 0.0

    def mm(self, a, b):         # multiply-add = 2
        self.n += 2.0 * a.shape[0] * a.shape[1] * (b.shape[1] if b.ndim > 1 else 1)
        return a @ b

    def add(self, k):
        self.n += k


def _relax(l, u):
    """ReLU relaxation per neuron: lower slope alpha, upper slope s, upper intercept b (h <= s z + b)."""
    act, dead = l >= 0, u <= 0
    unst = ~act & ~dead
    s = np.where(act, 1.0, 0.0)
    b = np.zeros_like(l)
    s[unst] = u[unst] / (u[unst] - l[unst])
    b[unst] = -s[unst] * l[unst]
    alpha = np.where(act, 1.0, 0.0)
    alpha[unst] = (u[unst] > -l[unst]).astype(float)
    return alpha, s, b


def _back(model, V, A, t_from, rel, fl, lower=True):
    """Linear bound of A . z_{t_from} (A: q x H) as a function of the simplex weights: returns coef (q, d).
    z_t = Wh h_{t-1} + Wi x_t, x_t = V[:, t] . lambda; h = relu(z) relaxed with rel[t-1]."""
    Wi, Wh, Wo = model
    q, d = A.shape[0], V.shape[0]
    coef = np.zeros((q, d))
    const = np.zeros(q)
    for t in range(t_from, -1, -1):          # z_t index t (0-based step)
        coef += np.outer(fl.mm(A, Wi[:, None])[:, 0], V[:, t])
        fl.add(2 * q * d)
        if t == 0:
            break
        Lam = fl.mm(A, Wh)                   # coefficients on h_{t-1}
        alpha, s, b = rel[t - 1]
        pos, neg = np.maximum(Lam, 0), np.minimum(Lam, 0)
        if lower:
            A = pos * alpha[None] + neg * s[None]
            const += neg @ b
        else:
            A = pos * s[None] + neg * alpha[None]
            const += pos @ b
        fl.add(6 * q * len(b))
    return coef + const[:, None]             # constant c -> c * sum(lambda) = c on the simplex


def crown(model, R, label, fl):
    """-> status: +1 certified correct, -1 certified wrong, 0 unknown."""
    Wi, Wh, Wo = model
    H, T = Wh.shape[0], R.shape[1]
    V = R                                    # d x T: row = vertex (generator), column = time
    rel = []
    I = np.eye(H)
    for t in range(T):                       # intermediate pre-activation bounds by back-substitution
        lo = _back(model, V, I, t, rel, fl, lower=True).min(1)
        hi = _back(model, V, I, t, rel, fl, lower=False).max(1)
        rel.append(_relax(lo, hi))
    riv = [k for k in range(Wo.shape[0]) if k != label]
    C = Wo[label][None] - Wo[riv]            # margins m_k = c_k . h_T
    # bound c . h_T: relax h_T = relu(z_T-1) then back-substitute
    alpha, s, b = rel[T - 1]
    pos, neg = np.maximum(C, 0), np.minimum(C, 0)
    AL, cL = pos * alpha[None] + neg * s[None], neg @ b
    AU, cU = pos * s[None] + neg * alpha[None], pos @ b
    lo = (_back(model, V, AL, T - 1, rel, fl, True) + cL[:, None]).min(1)
    hi = (_back(model, V, AU, T - 1, rel, fl, False) + cU[:, None]).max(1)
    if (lo > 0).all():
        return 1
    if (hi < 0).any():
        return -1
    return 0


def ibp(model, R, label, fl):
    """Interval bounds from the box hull of the simplex (E19-style)."""
    Wi, Wh, Wo = model
    T = R.shape[1]
    xl, xu = R.min(0), R.max(0)
    hl = hu = np.zeros(Wh.shape[0])
    Wp, Wn = np.maximum(Wh, 0), np.minimum(Wh, 0)
    for t in range(T):
        a, c = Wi * xl[t], Wi * xu[t]
        zl = Wp @ hl + Wn @ hu + np.minimum(a, c)
        zu = Wp @ hu + Wn @ hl + np.maximum(a, c)
        hl, hu = np.maximum(zl, 0), np.maximum(zu, 0)
        fl.add(4 * 2 * Wh.size)
    riv = [k for k in range(Wo.shape[0]) if k != label]
    C = Wo[label][None] - Wo[riv]
    lo = np.maximum(C, 0) @ hl + np.minimum(C, 0) @ hu
    hi = np.maximum(C, 0) @ hu + np.minimum(C, 0) @ hl
    fl.add(4 * C.size)
    if (lo > 0).all():
        return 1
    if (hi < 0).any():
        return -1
    return 0


def mass(R, fl):
    ops = M4.Ops()
    m = M4.mass4_rays(R, ops)
    fl.add(2 * ops.n if hasattr(ops, "n") else 2 * getattr(ops, "count", 600))
    return m


class Tree:
    def __init__(self, model, bound="crown"):
        self.model, self.bound = model, {"crown": crown, "ibp": ibp, "crown_fwd": crown_fwd}[bound]
        self.fl = Flops()
        self.nodes = []                       # dict(R, label, mass, status, kids, split)
        self.heap = []
        self.mass = {1: 0.0, -1: 0.0, 0: 0.0}
        for R, lab, key in roots(model[2].shape[0]):
            self._add(R, lab)
        self.root_key = {key: i for i, (_, _, key) in enumerate(roots(model[2].shape[0]))}

    def _add(self, R, label, m=None):
        f0 = self.fl.n
        m = mass(R, self.fl) if m is None else m
        f1 = self.fl.n
        st = self.bound(self.model, R, label, self.fl)
        self.flops_mass = getattr(self, "flops_mass", 0.0) + (f1 - f0)
        self.flops_bound = getattr(self, "flops_bound", 0.0) + (self.fl.n - f1)
        idx = len(self.nodes)
        self.nodes.append(dict(R=R, label=label, mass=m, status=st, kids=None, split=None))
        self.mass[st] += m
        if st == 0:
            heapq.heappush(self.heap, (-m, idx))
        return idx

    def step(self):
        _, idx = heapq.heappop(self.heap)
        nd = self.nodes[idx]
        R = nd["R"]
        G = R @ R.T
        iu = np.triu_indices(len(R), 1)
        k = np.argmin(G[iu])
        a, b = iu[0][k], iu[1][k]
        mvec = R[a] + R[b]
        mvec /= np.linalg.norm(mvec)
        RA, RB = R.copy(), R.copy()
        RA[b] = mvec                          # child A: lambda_a >= lambda_b
        RB[a] = mvec
        self.mass[0] -= nd["mass"]
        mA = mass(RA, self.fl)
        mB = nd["mass"] - mA                  # children partition the parent exactly
        ia = self._add(RA, nd["label"], mA)
        ib = self._add(RB, nd["label"], mB)
        nd["kids"], nd["split"], nd["status"] = (ia, ib), (a, b), None

    @property
    def n_leaves(self):
        return sum(1 for n in self.nodes if n["kids"] is None)

    def locate(self, x):
        """Leaf index of each point (diagnostic only)."""
        T = x.shape[1]
        srt = np.argsort(-x, 1)
        i, j = srt[:, 0], srt[:, 1]
        s = np.where(x[np.arange(len(x)), j] >= 0, 1, -1)
        cur = np.array([self.root_key[(a, b, c)] for a, b, c in zip(i, j, s)])
        while True:
            internal = np.array([self.nodes[c]["kids"] is not None for c in cur])
            if not internal.any():
                return cur
            for c in np.unique(cur[internal]):
                sel = np.where(cur == c)[0]
                nd = self.nodes[c]
                lam = np.linalg.solve(nd["R"].T, x[sel].T).T
                a, b = nd["split"]
                cur[sel] = np.where(lam[:, a] * np.linalg.norm(nd["R"][b]) >= lam[:, b] * np.linalg.norm(nd["R"][a]),
                                    nd["kids"][0], nd["kids"][1])


def crown_fwd(model, R, label, fl):
    """Cheaper variant: intermediate bounds by one forward symbolic (DeepPoly-forward) pass, sharing the work of
    all neurons and steps; back-substitution only for the final margins."""
    Wi, Wh, Wo = model
    H, T = Wh.shape[0], R.shape[1]
    d = R.shape[0]
    Wp, Wn = np.maximum(Wh, 0), np.minimum(Wh, 0)
    hL, hU = np.zeros((H, d)), np.zeros((H, d))
    rel = []
    for t in range(T):
        xin = np.outer(Wi, R[:, t])
        zL = fl.mm(Wp, hL) + fl.mm(Wn, hU) + xin
        zU = fl.mm(Wp, hU) + fl.mm(Wn, hL) + xin
        fl.add(4 * H * d)
        lo, hi = zL.min(1), zU.max(1)
        alpha, s, b = _relax(lo, hi)
        rel.append((alpha, s, b))
        hL = alpha[:, None] * zL
        hU = s[:, None] * zU + b[:, None]
        fl.add(4 * H * d)
    riv = [k for k in range(Wo.shape[0]) if k != label]
    C = Wo[label][None] - Wo[riv]
    alpha, s, b = rel[T - 1]
    pos, neg = np.maximum(C, 0), np.minimum(C, 0)
    AL, cL = pos * alpha[None] + neg * s[None], neg @ b
    AU, cU = pos * s[None] + neg * alpha[None], pos @ b
    lo = (_back(model, R, AL, T - 1, rel, fl, True) + cL[:, None]).min(1)
    hi = (_back(model, R, AU, T - 1, rel, fl, False) + cU[:, None]).max(1)
    if (lo > 0).all():
        return 1
    if (hi < 0).any():
        return -1
    return 0
