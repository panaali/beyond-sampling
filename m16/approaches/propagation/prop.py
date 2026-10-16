"""Distribution propagation for the accuracy of M_16,10 (no input sampling).

Key fact: the RNN has no biases, so within a fixed gate history the hidden state is an EXACT linear map of
the input, h_t = A_t x, x in R^10, and every event that matters (a ReLU gate being on, x_t beating the
running top-2, the final argmax being the true 2nd-largest position) is a homogeneous linear constraint on x.
So the input space splits into polyhedral cones, and accuracy is the Gaussian mass of the 'correct' cones.

We propagate a weighted set of branches. A branch = (top-2 bookkeeping p1,p2; gate history -> A_t;
a Gaussian N(mu, S) approximating x | branch constraints; mass w).  Each step:
  1. split on where x_t falls relative to x_p1, x_p2  (3 discrete outcomes)
  2. form preactivations a_i = (Wh A_t + Wi e_t)_i  (exact linear functionals of x)
  3. for each neuron, P(on) = Phi(a.mu / sqrt(a'Sa)); split the branch if both sides carry mass > eps,
     else take the majority side; condition the Gaussian on the chosen side (truncated-Gaussian moment
     matching = assumed density filtering)
  4. A_{t+1} = diag(gates) (Wh A_t + Wi e_t)
At the end, P(correct | branch) = P(logit_p2 > logit_j for all j), 9 more linear constraints (ADF, or EP).
Branches whose mass falls below eps are dropped (mass reported).  Optional merging moment-matches
branches with identical (p1, p2, gate pattern A_t).
"""
import numpy as np
from scipy.special import log_ndtr

T, D = 10, 10
LOG2PI_H = 0.5 * np.log(2 * np.pi)


def _lam(alpha):
    """phi(a)/Phi(a), stable."""
    return np.exp(-0.5 * alpha ** 2 - LOG2PI_H - log_ndtr(alpha))


class Counter:
    def __init__(self):
        self.flops = 0.0


def constrain(mu, S, a, sgn, C):
    """Condition N(mu,S) on sgn*(a.x) > 0 by moment matching. Returns Z (prob of the side). In-place."""
    Sa = np.einsum('bij,bj->bi', S, a)
    v = np.maximum((a * Sa).sum(1), 1e-300)
    m = (a * mu).sum(1)
    sv = np.sqrt(v)
    alpha = sgn * m / sv
    Z = np.exp(log_ndtr(alpha))
    lam = _lam(alpha)
    mu += (sgn * lam / sv)[:, None] * Sa
    S -= ((lam * (lam + alpha)) / v)[:, None, None] * Sa[:, :, None] * Sa[:, None, :]
    C.flops += len(mu) * (4 * D * D + 6 * D + 30)
    return Z


def prob_side(mu, S, a, C):
    Sa = np.einsum('bij,bj->bi', S, a)
    v = (a * Sa).sum(1)
    m = (a * mu).sum(1)
    C.flops += len(mu) * (2 * D * D + 4 * D + 20)
    return m, v


class Branches:
    fields = ('mu', 'S', 'A', 'w', 'p1', 'p2', 'cons')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def take(self, idx):
        return Branches(**{k: (getattr(self, k)[idx] if k != 'cons' else [self.cons[i] for i in idx])
                           for k in self.fields if hasattr(self, k)})

    @staticmethod
    def cat(bs):
        out = {}
        for k in Branches.fields:
            if not hasattr(bs[0], k):
                continue
            if k == 'cons':
                out[k] = [c for b in bs for c in b.cons]
            else:
                out[k] = np.concatenate([getattr(b, k) for b in bs])
        return Branches(**out)

    def __len__(self):
        return len(self.w)


def _add_cons(B, a, sgn, idx=None):
    if not hasattr(B, 'cons'):
        return
    rows = range(len(B)) if idx is None else idx
    for r in rows:
        B.cons[r] = B.cons[r] + [sgn[r] * a[r]]


def propagate(model, eps=1e-6, merge=True, keep_cons=False, final='adf', ep_iters=3, C=None,
              order='index'):
    Wi, Wh, Wo = model
    H = len(Wi)
    C = C or Counter()
    dropped = 0.0
    I = np.eye(D)
    B = Branches(mu=np.zeros((1, D)), S=I[None].copy(), A=np.zeros((1, H, D)), w=np.ones(1),
                 p1=np.array([-1]), p2=np.array([-1]))
    if keep_cons:
        B.cons = [[]]
    nb_hist = []
    for t in range(T):
        # ---- 1. rank split
        parts = []
        if t == 0:
            B.p1[:] = 0
            parts = [B]
        else:
            # outcome 0: x_t > x_p1
            b0 = B.take(np.arange(len(B)))
            a = np.zeros((len(B), D)); a[:, t] = 1; a[np.arange(len(B)), B.p1] -= 1
            z = constrain(b0.mu, b0.S, a, np.ones(len(B)), C); b0.w = b0.w * z
            if keep_cons: _add_cons(b0, a, np.ones(len(B)))
            b0.p2 = B.p1.copy(); b0.p1 = np.full(len(B), t)
            # outcome below p1
            b1 = B.take(np.arange(len(B)))
            z = constrain(b1.mu, b1.S, a, -np.ones(len(B)), C); b1.w = b1.w * z
            if keep_cons: _add_cons(b1, a, -np.ones(len(B)))
            if t == 1:
                b1.p2 = np.full(len(B), t)
                parts = [b0, b1]
            else:
                b2 = b1.take(np.arange(len(b1)))
                a2 = np.zeros((len(B), D)); a2[:, t] = 1; a2[np.arange(len(B)), B.p2] -= 1
                z = constrain(b1.mu, b1.S, a2, np.ones(len(B)), C); b1.w = b1.w * z
                if keep_cons: _add_cons(b1, a2, np.ones(len(B)))
                b1.p2 = np.full(len(B), t)
                z = constrain(b2.mu, b2.S, a2, -np.ones(len(B)), C); b2.w = b2.w * z
                if keep_cons: _add_cons(b2, a2, -np.ones(len(B)))
                parts = [b0, b1, b2]
        B = Branches.cat(parts)
        keep = B.w > eps; dropped += B.w[~keep].sum(); B = B.take(np.where(keep)[0])
        # ---- 2. preactivations
        P = np.einsum('ij,bjk->bik', Wh, B.A); P[:, :, t] += Wi[None]
        C.flops += len(B) * 2 * H * H * D
        G = np.zeros((len(B), H), bool)
        # ---- 3. gates
        neur = range(H)
        for i in neur:
            a = P[:, i, :]
            m, v = prob_side(B.mu, B.S, a, C)
            null = (a * a).sum(1) < 1e-20
            pon = np.where(null, 0.0, np.exp(log_ndtr(m / np.sqrt(np.maximum(v, 1e-300)))))
            split = (~null) & (B.w * np.minimum(pon, 1 - pon) > eps)
            maj = pon >= 0.5
            idx_s = np.where(split)[0]
            # copies for the minority side of split branches
            nb = B.take(idx_s); nP = P[idx_s]; nG = G[idx_s]
            sgn_main = np.where(split, 1.0, np.where(maj, 1.0, -1.0))
            sgn_new = -np.ones(len(idx_s))
            # main rows: split rows take 'on', unsplit take majority (null rows: no constraint, off)
            act = ~null
            if act.any():
                ia = np.where(act)[0]
                sub_mu, sub_S = B.mu[ia], B.S[ia]
                z = constrain(sub_mu, sub_S, a[ia], sgn_main[ia], C)
                B.mu[ia], B.S[ia] = sub_mu, sub_S
                lost = B.w[ia] * (1 - z); lost[split[ia]] = 0
                dropped += lost.sum()
                B.w[ia] = B.w[ia] * z
                G[ia, i] = sgn_main[ia] > 0
                if keep_cons:
                    for r in ia:
                        B.cons[r] = B.cons[r] + [sgn_main[r] * a[r]]
            if len(idx_s):
                z = constrain(nb.mu, nb.S, nP[:, i, :], sgn_new, C)
                nb.w = nb.w * z; nG[:, i] = False
                if keep_cons:
                    for k, r in enumerate(idx_s):
                        nb.cons[k] = nb.cons[k] + [-P[r, i]]
                B = Branches.cat([B, nb]); P = np.concatenate([P, nP]); G = np.concatenate([G, nG])
        B.A = P * G[:, :, None]
        keep = B.w > eps; dropped += B.w[~keep].sum(); B = B.take(np.where(keep)[0])
        if merge and not keep_cons:
            B = merge_branches(B, G[keep], C)
        nb_hist.append(len(B))
    # ---- 4. output
    tgt = B.p2
    L = np.einsum('jh,bhk->bjk', Wo, B.A)               # logits as functionals of x
    C.flops += len(B) * 2 * T * H * D
    if final == 'adf':
        zc = np.ones(len(B)); mu, S = B.mu.copy(), B.S.copy()
        for j in range(T):
            a = L[np.arange(len(B)), tgt] - L[:, j]
            act = tgt != j
            ia = np.where(act)[0]
            if len(ia):
                sm, sS = mu[ia], S[ia]
                zc[ia] *= constrain(sm, sS, a[ia], np.ones(len(ia)), C)
                mu[ia], S[ia] = sm, sS
        acc_b = zc
        Zb = B.w
    else:
        from ep import ep_logz
        acc_b = np.zeros(len(B)); Zb = np.zeros(len(B))
        for r in range(len(B)):
            cons = np.array(B.cons[r])
            outc = np.array([L[r, tgt[r]] - L[r, j] for j in range(T) if j != tgt[r]])
            lz_b = ep_logz(cons, ep_iters, C)
            lz_c = ep_logz(np.concatenate([cons, outc]), ep_iters, C)
            acc_b[r] = np.exp(lz_c - lz_b); Zb[r] = np.exp(lz_b)
    kept = Zb.sum()
    return dict(acc=float((Zb * acc_b).sum() / kept), acc_lower=float((Zb * acc_b).sum()),
                kept=float(kept), dropped=float(dropped), branches=nb_hist, flops=C.flops,
                n_final=len(B))


def merge_branches(B, G, C):
    """Moment-match branches with identical (p1, p2, A_t). A_t is determined by the gate history, which
    we key on via the bytes of A rounded (identical history -> identical A)."""
    keys = {}
    for r in range(len(B)):
        k = (int(B.p1[r]), int(B.p2[r]), B.A[r].round(10).tobytes())
        keys.setdefault(k, []).append(r)
    if len(keys) == len(B):
        return B
    rows = []
    mu, S, w = [], [], []
    for k, rs in keys.items():
        rows.append(rs[0])
        if len(rs) == 1:
            mu.append(B.mu[rs[0]]); S.append(B.S[rs[0]]); w.append(B.w[rs[0]]); continue
        ww = B.w[rs]; W = ww.sum(); pw = ww / W
        m = (pw[:, None] * B.mu[rs]).sum(0)
        dm = B.mu[rs] - m
        s = (pw[:, None, None] * (B.S[rs] + dm[:, :, None] * dm[:, None, :])).sum(0)
        mu.append(m); S.append(s); w.append(W)
        C.flops += len(rs) * 3 * D * D
    out = B.take(np.array(rows))
    out.mu = np.array(mu); out.S = np.array(S); out.w = np.array(w)
    return out
