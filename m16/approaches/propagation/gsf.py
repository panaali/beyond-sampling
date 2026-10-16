"""Gaussian-sum assumed-density filter over (hidden state, running top-2 values), conditioned on the
discrete bookkeeping (p1, p2) = positions of the running max and running 2nd max.

Continuous state per component: z = (h_1..h_16, m1, m2), approximated as Gaussian.  Each step:
  1. append x_t ~ N(0,1) (independent)  -> 19-dim Gaussian
  2. split into 3 discrete outcomes: x_t > m1 (new record), m2 < x_t < m1 (new 2nd), x_t < m2;
     each outcome = linear truncation, handled by truncated-Gaussian moment matching (prob = mass)
  3. optionally split on the sign of the n_split most ambiguous pre-activations (exact linear truncation)
  4. h' = ReLU(Wh h + Wi x_t): exact mean/covariance of a ReLU'd Gaussian (bivariate normal formulas),
     cross-covariance with (m1, m2, x_t) by Stein's lemma; select new (m1', m2')
  5. merge components with the same (p1, p2) down to K (moment matching)
At the last step every pre-activation whose sign is ambiguous (up to n_final neurons) is split exactly, so the
logits are an exact linear function of the Gaussian inside each piece; P(argmax logits = p2) is then a Gaussian
orthant probability over 9 margins (sequential moment matching).
Everything is deterministic and uses only the weights.  FLOPs are counted analytically.
"""
import numpy as np
from scipy.special import log_ndtr, ndtr
from relu_moments import relu_moments

T, H = 10, 16
IM1, IM2, IX = 16, 17, 18
LOG2PI_H = 0.5 * np.log(2 * np.pi)


class Counter:
    def __init__(self):
        self.flops = 0.0


def constrain(mu, S, a, C):
    """Condition N(mu,S) on a.z > 0 (moment matching). In place; returns probability."""
    Sa = np.einsum('bij,bj->bi', S, a)
    v = np.maximum((a * Sa).sum(1), 1e-300)
    m = (a * mu).sum(1)
    sv = np.sqrt(v)
    al = m / sv
    lZ = log_ndtr(al)
    lam = np.exp(-0.5 * al ** 2 - LOG2PI_H - lZ)
    mu += (lam / sv)[:, None] * Sa
    S -= ((lam * (lam + al)) / v)[:, None, None] * Sa[:, :, None] * Sa[:, None, :]
    d = mu.shape[1]
    C.flops += len(mu) * (4 * d * d + 6 * d + 40)
    return np.exp(lZ)


class Comp:
    def __init__(self, mu, S, w, p1, p2):
        self.mu, self.S, self.w, self.p1, self.p2 = mu, S, w, p1, p2

    def take(self, i):
        return Comp(self.mu[i].copy(), self.S[i].copy(), self.w[i].copy(), self.p1[i].copy(), self.p2[i].copy())

    @staticmethod
    def cat(cs):
        cs = [c for c in cs if len(c.w)]
        return Comp(*[np.concatenate([getattr(c, k) for c in cs]) for k in ('mu', 'S', 'w', 'p1', 'p2')])

    def __len__(self):
        return len(self.w)


def merge(c, K, C, rule='mass'):
    """Reduce to at most K components per (p1, p2) by moment matching."""
    key = c.p1 * 100 + c.p2
    order = np.lexsort((-c.w, key))
    key_s = key[order]
    starts = np.r_[0, np.where(np.diff(key_s))[0] + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    grp = np.minimum(rank, K - 1)                     # within key: top K-1 kept, rest pooled
    gid_s = np.unique(np.stack([key_s, grp], 1), axis=0, return_inverse=True)[1].ravel()
    gid = np.empty(len(order), int); gid[order] = gid_s
    G = gid.max() + 1
    W = np.bincount(gid, c.w, G)
    pw = c.w / W[gid]
    d = c.mu.shape[1]
    mu = np.zeros((G, d)); np.add.at(mu, gid, pw[:, None] * c.mu)
    dm = c.mu - mu[gid]
    S = np.zeros((G, d, d)); np.add.at(S, gid, pw[:, None, None] * (c.S + dm[:, :, None] * dm[:, None, :]))
    first = np.zeros(G, int); first[gid[::-1]] = np.arange(len(gid))[::-1]
    C.flops += len(c) * 3 * d * d
    return Comp(mu, S, W, c.p1[first], c.p2[first])


def preact_rows(Wi, Wh):
    M = np.zeros((H, 19)); M[:, :H] = Wh; M[:, IX] = Wi
    return M


def split_gates(c, M, n_split, eps, C, min_amb=0.0):
    """Split each component on the signs of its n_split most ambiguous pre-activations."""
    if n_split == 0:
        return c
    mu_u = c.mu @ M.T
    var_u = np.einsum('ij,bjk,ik->bi', M, c.S, M)
    C.flops += len(c) * (2 * H * 19 * 19 + 2 * H * 19)
    pon = ndtr(mu_u / np.sqrt(np.maximum(var_u, 1e-300)))
    amb = np.minimum(pon, 1 - pon)
    amb[var_u < 1e-18] = 0
    order = np.argsort(-amb, 1)[:, :n_split]
    for k in range(n_split):
        idx = order[:, k]
        a_k = amb[np.arange(len(c)), idx]
        do = (a_k * c.w > eps) & (a_k > min_amb)
        if not do.any():
            continue
        di = np.where(do)[0]
        on = c.take(di); off = c.take(di)
        a = M[idx[di]]
        on.w = on.w * constrain(on.mu, on.S, a, C)
        off.w = off.w * constrain(off.mu, off.S, -a, C)
        rest = c.take(np.where(~do)[0])
        c = Comp.cat([rest, on, off])
        order = np.concatenate([order[~do], order[di], order[di]])
        amb = np.concatenate([amb[~do], amb[di], amb[di]])
    return c


def relu_step(c, M, sel, C):
    """z19 -> z18: h' = relu(M z), (m1', m2') = z[sel]."""
    B = len(c)
    mu_u = c.mu @ M.T
    MS = np.einsum('ij,bjk->bik', M, c.S)              # Cov(u, z)  (B,16,19)
    Cu = np.einsum('bik,jk->bij', MS, M)
    C.flops += B * (2 * H * 19 * 19 + 2 * H * H * 19)
    m, cov, Pa = relu_moments(mu_u, Cu)
    C.flops += B * (H * H * 120 + H * 60)              # pairwise bivariate-normal terms (Owen's T ~ 60 flops)
    zs = sel                                           # (B,2) indices of new m1,m2 in z19
    mu = np.zeros((B, 18)); S = np.zeros((B, 18, 18))
    mu[:, :H] = m
    mu[:, 16:] = np.take_along_axis(c.mu, zs, 1)
    S[:, :H, :H] = cov
    Szz = c.S[np.arange(B)[:, None, None], zs[:, :, None], zs[:, None, :]]
    S[:, 16:, 16:] = Szz
    cross = Pa[:, :, None] * np.take_along_axis(MS, np.broadcast_to(zs[:, None, :], (B, H, 2)), 2)
    S[:, :H, 16:] = cross; S[:, 16:, :H] = cross.transpose(0, 2, 1)
    return Comp(mu, S, c.w, c.p1, c.p2)


def rank_split(c, t, C):
    """Augmented 19-dim components -> list of (component, sel) for the 3 outcomes of x_t."""
    B = len(c)
    if t == 0:
        out = c.take(np.arange(B)); out.p1[:] = 0; out.p2[:] = -1
        return [(out, np.tile([IX, IM2], (B, 1)))]
    a1 = np.zeros((B, 19)); a1[:, IX] = 1; a1[:, IM1] = -1      # x_t - m1
    rec = c.take(np.arange(B)); rec.w = rec.w * constrain(rec.mu, rec.S, a1, C)
    rec.p2 = c.p1.copy(); rec.p1 = np.full(B, t)
    below1 = c.take(np.arange(B)); below1.w = below1.w * constrain(below1.mu, below1.S, -a1, C)
    if t == 1:
        below1.p2 = np.full(B, t)
        return [(rec, np.tile([IX, IM1], (B, 1))), (below1, np.tile([IM1, IX], (B, 1)))]
    a2 = np.zeros((B, 19)); a2[:, IX] = 1; a2[:, IM2] = -1      # x_t - m2
    mid = below1.take(np.arange(B)); mid.w = mid.w * constrain(mid.mu, mid.S, a2, C); mid.p2 = np.full(B, t)
    low = below1; low.w = low.w * constrain(low.mu, low.S, -a2, C)
    return [(rec, np.tile([IX, IM1], (B, 1))), (mid, np.tile([IM1, IX], (B, 1))),
            (low, np.tile([IM1, IM2], (B, 1)))]


def final_prob(c, M, Wo, n_final, eps, C):
    """P(argmax logits == p2) for each component of the augmented last-step state, after exact gate splits."""
    c = split_gates(c, M, n_final, eps, C)
    B = len(c)
    mu_u = c.mu @ M.T
    var_u = np.einsum('ij,bjk,ik->bi', M, c.S, M)
    # gate: split neurons are exactly determined (truncated); others use the more likely side
    g = (mu_u > 0).astype(float)
    C.flops += B * (2 * H * 19 * 19)
    Lz = np.einsum('jh,bh,hk->bjk', Wo, g, M)            # logits as linear functionals of z19 (B,10,19)
    C.flops += B * 2 * T * H * 19
    tgt = c.p2
    z = np.ones(B); mu, S = c.mu.copy(), c.S.copy()
    # constrain in order of increasing margin probability (most binding first)
    a_all = Lz[np.arange(B), tgt][:, None, :] - Lz            # (B,10,19)
    mm = np.einsum('bjk,bk->bj', a_all, mu)
    vv = np.einsum('bjk,bkl,bjl->bj', a_all, S, a_all)
    pr = np.where(np.arange(T)[None] == tgt[:, None], np.inf, mm / np.sqrt(np.maximum(vv, 1e-300)))
    order = np.argsort(pr, 1)
    for k in range(T - 1):
        j = order[:, k]
        z *= constrain(mu, S, a_all[np.arange(B), j], C)
    return c.w, z


def gsf(model, K=1, n_split=0, n_final=4, eps=1e-7, C=None):
    Wi, Wh, Wo = model
    C = C or Counter()
    M = preact_rows(Wi, Wh)
    c = Comp(np.zeros((1, 18)), np.zeros((1, 18, 18)), np.ones(1), np.array([-1]), np.array([-1]))
    sizes = []
    for t in range(T):
        B = len(c)
        mu = np.concatenate([c.mu, np.zeros((B, 1))], 1)
        S = np.zeros((B, 19, 19)); S[:, :18, :18] = c.S; S[:, IX, IX] = 1
        aug = Comp(mu, S, c.w, c.p1, c.p2)
        pieces = rank_split(aug, t, C)
        if t == T - 1:
            ws, zs = [], []
            for comp, sel in pieces:
                comp.p2 = comp.p2  # target = global 2nd position
                w, z = final_prob(comp, M, Wo, n_final, eps, C)
                ws.append(w); zs.append(z)
            w = np.concatenate(ws); z = np.concatenate(zs)
            sizes.append(len(w))
            return dict(acc=float((w * z).sum() / w.sum()), mass=float(w.sum()), flops=C.flops, sizes=sizes)
        new = []
        for comp, sel in pieces:
            keep = comp.w > eps
            comp = comp.take(np.where(keep)[0]); sel = sel[keep]
            if not len(comp):
                continue
            n0 = len(comp)
            comp2 = split_gates(comp, M, n_split, eps, C)
            sel2 = np.tile(sel[0], (len(comp2), 1))  # sel identical within a piece
            new.append(relu_step(comp2, M, sel2, C))
        c = merge(Comp.cat(new), K, C)
        sizes.append(len(c))
