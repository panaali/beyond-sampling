"""Gaussian-sum filter keyed by mechanism state (p1, p2, current gate pattern g).

Components: z = (h_1..h_16, m1, m2) ~ Gaussian, with discrete key (p1, p2, g_t).  Each step:
  1. append x_t ~ N(0,1); split on its rank vs (m1, m2)  (3 outcomes, truncated-Gaussian moment matching)
  2. pre-activations u = Wh h + Wi x_t are jointly Gaussian.  Enumerate gate patterns by deciding neurons
     in order of decreasing ambiguity.  The probability of each sign is P(s_j u_j > 0 | s_k u_k > 0), where k is
     the already-decided neuron most correlated with j (exact bivariate normal, a tree approximation to the
     orthant probability, which handles the near-collinear gates exactly).  A sign whose branch mass is
     < eps is pruned (its mass is dropped and reported).  Split branches are conditioned by truncated-Gaussian
     moment matching.
  3. inside a gate pattern, h' = diag(g) u is EXACTLY linear in z, so the new Gaussian is exact given the
     (approximate) truncated input Gaussian.
  4. merge components with identical key; keep the K heaviest patterns per (p1,p2), moment-merge the rest into
     one 'rest' component per (p1,p2).
Final step: same gate enumeration, then logits = Wo diag(g) u are linear in z; P(argmax = p2) by sequential
moment matching over the 9 margins (most binding first).
"""
import numpy as np
from scipy.special import log_ndtr, ndtr
from relu_moments import Phi2

T, H = 10, 16
IM1, IM2, IX = 16, 17, 18
LOG2PI_H = 0.5 * np.log(2 * np.pi)
FL_PHI2 = 80   # flop charge for one bivariate normal cdf


class Counter:
    def __init__(self):
        self.flops = 0.0


def constrain(mu, S, a, C):
    Sa = np.einsum('bij,bj->bi', S, a)
    v = np.maximum((a * Sa).sum(1), 1e-300)
    m = (a * mu).sum(1)
    sv = np.sqrt(v); al = m / sv
    lZ = log_ndtr(al)
    lam = np.exp(-0.5 * al ** 2 - LOG2PI_H - lZ)
    mu += (lam / sv)[:, None] * Sa
    S -= ((lam * (lam + al)) / v)[:, None, None] * Sa[:, :, None] * Sa[:, None, :]
    d = mu.shape[1]
    C.flops += len(mu) * (4 * d * d + 6 * d + 40)
    return np.exp(lZ)


class Comp:
    keys = ('mu', 'S', 'w', 'p1', 'p2', 'g')

    def __init__(self, **kw):
        for k in self.keys:
            setattr(self, k, kw[k])

    def take(self, i):
        return Comp(**{k: getattr(self, k)[i].copy() for k in self.keys})

    @staticmethod
    def cat(cs):
        cs = [c for c in cs if len(c.w)]
        return Comp(**{k: np.concatenate([getattr(c, k) for c in cs]) for k in Comp.keys})

    def __len__(self):
        return len(self.w)


def enumerate_gates(c, M, eps, C, dropped):
    """c: augmented 19-dim comps (g field ignored). Returns comps with gate pattern g (B,16) bool and the
    (truncated) 19-dim Gaussian; weights multiplied by the pattern probability."""
    B = len(c)
    mu_u = c.mu @ M.T
    MS = np.einsum('ij,bjk->bik', M, c.S)
    Cu = np.einsum('bik,jk->bij', MS, M)
    C.flops += B * (2 * H * 19 * 19 + 2 * H * H * 19 + 2 * H * 19)
    sd = np.sqrt(np.maximum(np.einsum('bii->bi', Cu), 1e-300))
    al = mu_u / sd
    R = Cu / (sd[:, :, None] * sd[:, None, :])
    null = np.einsum('bii->bi', Cu) < 1e-14
    amb = np.minimum(ndtr(al), ndtr(-al)); amb[null] = -1
    order = np.argsort(-amb, 1)                      # decide most ambiguous first
    # state carried per branch
    mu, S, w = c.mu.copy(), c.S.copy(), c.w.copy()
    src = np.arange(B)                               # which parent (for mu_u, R, order)
    sgn = np.zeros((B, H))                           # +1 on, -1 off, 0 undecided
    for k in range(H):
        j = order[src, k]
        n = len(w)
        aj = al[src, j]
        isnull = null[src, j]
        if k == 0:
            pon = ndtr(aj)
        else:
            decided = order[src, :k]                 # (n,k)
            r = R[src[:, None], j[:, None], decided]  # corr with decided
            kk = np.argmax(np.abs(r), 1)
            i = decided[np.arange(n), kk]; r0 = r[np.arange(n), kk]; rr = np.clip(r0, -0.999, 0.999)
            si = sgn[np.arange(n), i]; ai = al[src, i]
            # P(u_j>0 | s_i u_i > 0) = P(u_j>0, s_i u_i>0) / P(s_i u_i>0)
            # with U=-u standardised: P(U_j < aj, s_i... ) -> use Phi2(aj, si*ai, si*rr)
            num = Phi2(aj, si * ai, si * rr)
            den = ndtr(si * ai)
            pon = np.clip(num / np.maximum(den, 1e-300), 0, 1)
            col = np.abs(r0) > 0.999                     # (near-)collinear with a decided gate: implied
            pon = np.where(col, (si * np.sign(r0) > 0).astype(float), pon)
            C.flops += n * (FL_PHI2 + 2 * k)
        pon = np.where(isnull, 0.0, pon)
        keep_on = w * pon > eps
        keep_off = w * (1 - pon) > eps
        neither = ~keep_on & ~keep_off
        keep_on |= neither & (pon >= 0.5); keep_off |= neither & (pon < 0.5)
        both = keep_on & keep_off & ~isnull
        # a side whose mass is < eps is absorbed into the other side (mass conserved, no conditioning)
        # rows that go on only / off only: no conditioning (implied), weight *= p
        a = M[j]
        # build on-branch and off-branch
        on_idx = np.where(keep_on & ~isnull)[0]
        off_idx = np.where((keep_off & ~isnull) | isnull)[0]
        out = []
        for idx, s in ((on_idx, 1.0), (off_idx, -1.0)):
            if not len(idx):
                continue
            m_, S_ = mu[idx].copy(), S[idx].copy()
            bi = both[idx]
            p = np.where(bi, pon[idx] if s > 0 else 1 - pon[idx], 1.0)
            if bi.any():
                ii = np.where(bi)[0]
                sm, sS = m_[ii], S_[ii]
                constrain(sm, sS, s * a[idx[ii]], C)
                m_[ii], S_[ii] = sm, sS
            sg = sgn[idx].copy(); sg[np.arange(len(idx)), j[idx]] = np.where(isnull[idx], -1, s)
            out.append((m_, S_, w[idx] * p, src[idx], sg))
        mu = np.concatenate([o[0] for o in out]); S = np.concatenate([o[1] for o in out])
        w = np.concatenate([o[2] for o in out]); src = np.concatenate([o[3] for o in out])
        sgn = np.concatenate([o[4] for o in out])
    return Comp(mu=mu, S=S, w=w, p1=c.p1[src], p2=c.p2[src], g=sgn > 0)


def rank_split(c, t, C):
    B = len(c)
    if t == 0:
        out = c.take(np.arange(B)); out.p1[:] = 0; out.p2[:] = -1
        return [(out, (IX, IM2))]
    a1 = np.zeros((B, 19)); a1[:, IX] = 1; a1[:, IM1] = -1
    rec = c.take(np.arange(B)); rec.w = rec.w * constrain(rec.mu, rec.S, a1, C)
    rec.p2 = c.p1.copy(); rec.p1 = np.full(B, t)
    below = c.take(np.arange(B)); below.w = below.w * constrain(below.mu, below.S, -a1, C)
    if t == 1:
        below.p2 = np.full(B, t)
        return [(rec, (IX, IM1)), (below, (IM1, IX))]
    a2 = np.zeros((B, 19)); a2[:, IX] = 1; a2[:, IM2] = -1
    mid = below.take(np.arange(B)); mid.w = mid.w * constrain(mid.mu, mid.S, a2, C); mid.p2 = np.full(B, t)
    low = below; low.w = low.w * constrain(low.mu, low.S, -a2, C)
    return [(rec, (IX, IM1)), (mid, (IM1, IX)), (low, (IM1, IM2))]


def linear_step(c, M, sel, C):
    """z19 -> z18 with h' = g*u (exact linear), (m1', m2') = z[sel]."""
    B = len(c)
    L = np.zeros((B, 18, 19))
    L[:, :H, :] = c.g[:, :, None] * M[None]
    L[:, 16, sel[0]] = 1; L[:, 17, sel[1]] = 1
    mu = np.einsum('bij,bj->bi', L, c.mu)
    S = np.einsum('bij,bjk,blk->bil', L, c.S, L)
    C.flops += B * (2 * 18 * 19 * 19 + 2 * 18 * 18 * 19)
    return Comp(mu=mu, S=S, w=c.w, p1=c.p1, p2=c.p2, g=c.g)


def merge(c, K, C):
    """Merge identical keys; keep K heaviest gate patterns per (p1,p2); pool the rest per (p1,p2)."""
    gb = np.packbits(c.g, axis=1).view(np.uint16)[:, 0].astype(np.int64)
    pk = (c.p1 + 1) * 11 + (c.p2 + 1)
    full = pk * 65536 + gb
    uk, inv = np.unique(full, return_inverse=True); inv = inv.ravel()
    W = np.bincount(inv, c.w)
    # rank patterns within (p1,p2)
    upk = uk // 65536
    o = np.lexsort((-W, upk))
    starts = np.r_[0, np.where(np.diff(upk[o]))[0] + 1]
    rank = np.empty(len(uk), int)
    rank[o] = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)]))
    grp_key = np.where(rank < K, uk, -(upk + 1))   # pooled group id per (p1,p2)
    gk, gid_u = np.unique(grp_key, return_inverse=True)
    gid = gid_u.ravel()[inv]
    G = len(gk)
    Wg = np.bincount(gid, c.w, G)
    pw = c.w / Wg[gid]
    d = c.mu.shape[1]
    mu = np.zeros((G, d)); np.add.at(mu, gid, pw[:, None] * c.mu)
    dm = c.mu - mu[gid]
    S = np.zeros((G, d, d)); np.add.at(S, gid, pw[:, None, None] * (c.S + dm[:, :, None] * dm[:, None, :]))
    first = np.full(G, len(gid)); np.minimum.at(first, gid, np.arange(len(gid)))
    C.flops += len(c) * 3 * d * d
    g = c.g[first].copy()
    pooled = gk < 0
    return Comp(mu=mu, S=S, w=Wg, p1=c.p1[first], p2=c.p2[first], g=g)


def final_prob(c, M, Wo, C):
    B = len(c)
    Lz = np.einsum('jh,bh,hk->bjk', Wo, c.g.astype(float), M)
    C.flops += B * 2 * T * H * 19
    tgt = c.p2
    a_all = Lz[np.arange(B), tgt][:, None, :] - Lz
    mu, S = c.mu.copy(), c.S.copy()
    mm = np.einsum('bjk,bk->bj', a_all, mu)
    vv = np.einsum('bjk,bkl,bjl->bj', a_all, S, a_all)
    C.flops += B * T * (2 * 19 * 19 + 4 * 19)
    pr = np.where(np.arange(T)[None] == tgt[:, None], np.inf, mm / np.sqrt(np.maximum(vv, 1e-300)))
    order = np.argsort(pr, 1)
    z = np.ones(B)
    for k in range(T - 1):
        j = order[:, k]
        z *= constrain(mu, S, a_all[np.arange(B), j], C)
    return z


def gsf2(model, K=8, eps=1e-6, C=None, renorm=True):
    Wi, Wh, Wo = model
    C = C or Counter()
    M = np.zeros((H, 19)); M[:, :H] = Wh; M[:, IX] = Wi
    c = Comp(mu=np.zeros((1, 18)), S=np.zeros((1, 18, 18)), w=np.ones(1), p1=np.array([-1]),
             p2=np.array([-1]), g=np.zeros((1, H), bool))
    sizes, dropped = [], [0.0]
    for t in range(T):
        B = len(c)
        mu = np.concatenate([c.mu, np.zeros((B, 1))], 1)
        S = np.zeros((B, 19, 19)); S[:, :18, :18] = c.S; S[:, IX, IX] = 1
        aug = Comp(mu=mu, S=S, w=c.w, p1=c.p1, p2=c.p2, g=c.g)
        new, fin = [], []
        for comp, sel in rank_split(aug, t, C):
            keep = comp.w > eps; dropped[0] += comp.w[~keep].sum()
            comp = comp.take(np.where(keep)[0])
            if not len(comp):
                continue
            gc = enumerate_gates(comp, M, eps, C, dropped)
            if t == T - 1:
                fin.append((gc.w, final_prob(gc, M, Wo, C), gc.p1 * 11 + gc.p2))
            else:
                new.append(linear_step(gc, M, sel, C))
        if t == T - 1:
            w = np.concatenate([f[0] for f in fin]); z = np.concatenate([f[1] for f in fin])
            if renorm:
                pk = np.concatenate([f[2] for f in fin])
                u, inv = np.unique(pk, return_inverse=True); inv = inv.ravel()
                w = w / np.bincount(inv, w)[inv] / 90.0
            sizes.append(len(w))
            return dict(acc=float((w * z).sum() / w.sum()), mass=float(w.sum()), dropped=dropped[0],
                        flops=C.flops, sizes=sizes)
        c = merge(Comp.cat(new), K, C)
        if renorm:   # exact, model-free: P(p1, p2 after step t) = 1 / (t (t+1)) for every ordered pair
            pk = (c.p1 + 1) * 11 + (c.p2 + 1)
            u, inv = np.unique(pk, return_inverse=True); inv = inv.ravel()
            tot = np.bincount(inv, c.w)
            exact = 1.0 if t == 0 else 1.0 / (t * (t + 1))
            c.w = c.w * exact / tot[inv]
        sizes.append(len(c))
