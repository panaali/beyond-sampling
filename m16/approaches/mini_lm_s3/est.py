"""T446 rare-event estimators for B_tau = P(max_t p_illegal > tau) on dyck2 (prefix tree).

Node = prefix of length t (t = 0..L-1); its value phi = p_illegal at position t (which depends only on the
prefix). The last token y[L-1] never matters. Cost is charged per node computed (node_cost; a full sequence
= 1 FPE, so plain MC costs 1 FPE per sample). All estimators are unbiased (up to the stated caveats) and
sampling-based; top-risk enumeration has an exact (deductive) head.
"""
import heapq
import math

import numpy as np
import torch

from s3lib import Dyck2, logits_of, stack_info

Q = 0.5


class Cost:
    def __init__(self, model, L):
        f = [model.flops(t) for t in range(L + 1)]
        self.c = np.array([(f[t + 1] - f[t]) / f[L] for t in range(L)])  # cost of node at position t
        self.total = 0.0

    def add(self, t, n=1):
        self.total += float(self.c[t]) * n

    def add_range(self, t0, t1, n=1):  # positions t0..t1-1
        self.total += float(self.c[t0:t1].sum()) * n


@torch.no_grad()
def phi_at(model, lang, y, t, chunk=8192):
    """p_illegal at position t for prefixes y[:, :t] (y is (N, >=t)). Returns (N,)."""
    N = len(y)
    x = np.empty((N, t + 1), np.int64)
    x[:, 0] = lang.BOS
    x[:, 1:] = y[:, :t]
    out = np.empty(N)
    for i in range(0, N, chunk):
        lg = model(torch.from_numpy(x[i:i + chunk]))[:, -1].double()
        p = torch.softmax(lg, -1).numpy()
        yy = y[i:i + chunk, :t]
        d = np.zeros(len(yy), np.int64); top = np.full(len(yy), -1)
        if t > 0:
            dep, tp, _ = stack_info(np.concatenate([yy, np.zeros((len(yy), 1), np.int64)], 1))
            d, top = dep[:, t], tp[:, t]
        wrong = np.where(top == 0, 3, 2)
        out[i:i + chunk] = np.where(d > 0, p[np.arange(len(p)), wrong], p[:, 2] + p[:, 3])
    return out


def phi_all(model, lang, y, chunk=8192):
    """(N, L) p_illegal at every position for full sequences y (N, L)."""
    lg = torch.from_numpy(logits_of(model, lang, y, chunk=chunk))
    p = torch.softmax(lg, -1).numpy()
    dep, top, _ = stack_info(y)
    wrong = np.where(top == 0, 3, 2)
    pw = np.take_along_axis(p, wrong[..., None], -1)[..., 0]
    return np.where(dep > 0, pw, p[..., 2] + p[..., 3])


# ----------------------------------------------------------------------------- language walker
class Walk:
    """Vectorised Dyck2 prefixes. Actions: 0 = open same type as top (type 0 at d = 0), 1 = open other, 2 = close."""

    def __init__(self, N, L):
        self.N, self.L = N, L
        self.y = np.zeros((N, L), np.int64)
        self.st = np.zeros((N, L + 1), np.int64)
        self.d = np.zeros(N, np.int64)
        self.sp = np.zeros((N, L + 1), np.int64)  # position of each stack entry's opener
        self.ncl = np.zeros(N, np.int64)  # consecutive closes just made
        self.t = 0

    def top(self):
        return np.where(self.d > 0, self.st[np.arange(self.N), np.maximum(self.d - 1, 0)], 0)

    def law(self):
        p = np.empty((self.N, 3))
        z = self.d == 0
        p[:, 0] = np.where(z, .5, Q / 2); p[:, 1] = np.where(z, .5, Q / 2); p[:, 2] = np.where(z, 0, 1 - Q)
        return p

    def step(self, a):
        idx = np.arange(self.N)
        tp = self.top()
        tok = np.where(a == 0, tp, np.where(a == 1, 1 - tp, 2 + tp))
        self.y[:, self.t] = tok
        op = a < 2
        self.st[idx[op], self.d[op]] = tok[op]
        self.sp[idx[op], self.d[op]] = self.t
        self.d = self.d + np.where(op, 1, -1)
        self.ncl = np.where(op, 0, self.ncl + 1)
        self.t += 1

    def features(self):
        """Proposal features at the current position."""
        N, d = self.N, self.d
        tp = self.top()
        F = [np.ones(N)]
        for k in range(1, 11):
            F.append((np.minimum(d, 10) == k).astype(float))
        F.append(np.full(N, self.t / self.L))
        F.append(tp.astype(float) * (d > 0))
        # fraction of stack entries equal to the top type, and whether the 2nd entry equals the top
        cols = np.arange(self.L + 1)[None, :]
        live = cols < d[:, None]
        same = ((self.st == tp[:, None]) & live).sum(1)
        F.append(np.where(d > 0, same / np.maximum(d, 1), 0.))
        sec = self.st[np.arange(N), np.maximum(d - 2, 0)]
        F.append(((d > 1) & (sec == tp)).astype(float))
        F.append(((d > 1) & (sec != tp)).astype(float))
        tl = self.t / self.L
        F.append(np.full(N, tl * tl))
        F.append(np.minimum(d, 12) * tl / 12)
        tpos = self.sp[np.arange(N), np.maximum(d - 1, 0)]
        F.append(np.where(d > 0, (self.t - tpos) / self.L, 0.))  # distance to the top's opener
        F.append(np.minimum(self.ncl, 6) / 6)
        F.append((self.ncl > 0).astype(float))
        # first index where the stack changes type below the top (length of the top same-type run)
        run = np.ones(N)
        for k in range(2, 8):
            e = self.st[np.arange(N), np.maximum(d - k, 0)]
            run += ((d >= k) & (e == tp) & (run == k - 1))
        F.append(np.minimum(run, 7) / 7 * (d > 0))
        return np.stack(F, 1)


def sample_law(N, L, rng):
    w = Walk(N, L)
    for t in range(L):
        p = w.law()
        u = rng.random(N)
        a = (u > p[:, 0]).astype(np.int64) + (u > p[:, 0] + p[:, 1])
        w.step(a)
    return w.y


# ----------------------------------------------------------------------------- plain MC
def plain_mc(model, lang, L, tau, budget, rng):
    N = int(budget)
    hits = 0
    for i in range(0, N, 50000):
        n = min(50000, N - i)
        y = sample_law(n, L, rng)
        hits += int((phi_all(model, lang, y)[:, :L].max(1) > tau).sum())
    return hits / N, float(N)


# ----------------------------------------------------------------------------- AMS (multilevel splitting)
def close_all(y, t, L):
    """Sequences y[:, :t] followed by the closers of the whole stack (then openers as filler)."""
    N = len(y)
    dep, top, _ = stack_info(y)
    Z = np.zeros((N, L), np.int64)
    Z[:, :t] = y[:, :t]
    w = Walk(N, L)
    for u in range(t):
        tok = y[:, u]; tp = w.top()
        w.step(np.where(tok >= 2, 2, np.where(tok == tp, 0, 1)))
    d0 = w.d.copy()
    for u in range(t, L):
        w.step(np.where(w.d > 0, 2, 0))
    return w.y, d0


def raw_scores(model, lang, y, tau, L, look, cost=None, t_from=0):
    """(N, L) raw per-position values: phi (look = False) or the discounted close-all lookahead
    max_j phi_{t+j} 0.5^j (look = True; phi_t included at j = 0). Also returns phi."""
    ph = phi_all(model, lang, y)
    R = ph.copy()
    if look:
        for t in range(t_from, L - 1):
            Z, d0 = close_all(y, t, L)
            pz = phi_all(model, lang, Z)
            for j in range(1, L - t):
                R[:, t] = np.maximum(R[:, t], np.where(j <= d0, np.minimum(pz[:, t + j], tau) * 0.5 ** j, 0))
            if cost is not None:
                for j in range(1, L - t):
                    cost.add(t + j, int((d0 >= j).sum()))
    return R, ph


class Rarity:
    """Per-position rarity transform r_t(v) = -log((#{pilot_t >= v} + .5) / (n + 1)), extended beyond the
    pilot's max by the logit difference; mapped to [0, 1) by r / (r + 10)."""

    def __init__(self, R):
        self.s = np.sort(R, 0)
        self.n = len(R)
        lg = lambda v: np.log(np.clip(v, 1e-300, 1)) - np.log1p(-np.clip(v, 0, 1 - 1e-12))
        self.lg = lg
        self.mx = self.s[-1]

    def __call__(self, R):
        out = np.empty_like(R)
        for t in range(R.shape[1]):
            ge = self.n - np.searchsorted(self.s[:, t], R[:, t], side="left")
            r = -np.log((ge + 0.5) / (self.n + 1))
            ext = np.log(self.n + 1) + np.maximum(self.lg(R[:, t]) - self.lg(self.mx[t]), 0)
            out[:, t] = np.where(R[:, t] > self.mx[t], ext, r)
        return 0.999 * out / (out + 10.0)


def ams(model, lang, L, tau, budget, rng, frac=0.1, N=None, look=False, f_pilot=0.1, imp=None, cost0=0.0):
    """Generalised adaptive multilevel splitting (Brehier et al. 2016) over prefixes. Score of a path =
    max_t xi_t, xi_t = 2 if p_illegal_t > tau (event), else the per-position rarity of the raw value
    (running p_illegal, or the close-all lookahead), calibrated on a separate charged pilot sample. Kills every
    particle at or below the current level (unbiased with ties); final level 1. Killed particles branch from a
    random survivor at the first position whose xi exceeds the level; the rest is resampled from the law."""
    cost = Cost(model, L)
    cost.total = cost0
    per = 1.0 if not look else 4.0
    if imp is not None:
        def raw_scores(model, lang, yy, tau, L, look, cost=None, t_from=0):  # noqa: F811
            return imp(yy), phi_all(model, lang, yy)
    else:
        raw_scores = globals()["raw_scores"]
    n_pil = max(200, int(f_pilot * budget / per))
    yp = sample_law(n_pil, L, rng)
    Rp, _ = raw_scores(model, lang, yp, tau, L, look, cost)
    cost.add_range(0, L, n_pil)
    tr = Rarity(Rp)
    if N is None:
        N = max(200, int(budget / (20 * per)))

    def xi(yy, t_from=0):
        R, ph = raw_scores(model, lang, yy, tau, L, look, cost, t_from)
        return np.where(ph > tau, 2.0, tr(R))
    y = sample_law(N, L, rng)
    X = xi(y)
    cost.add_range(0, L, N)
    logp = 0.0
    S = X.max(1)
    while True:
        K = max(1, int(frac * N))
        lvl = np.partition(S, K - 1)[K - 1]
        if lvl >= 1.0:
            break
        kill = np.nonzero(S <= lvl)[0]
        surv = np.nonzero(S > lvl)[0]
        if len(surv) == 0:
            return 0.0, cost.total
        logp += math.log(1 - len(kill) / N)
        src = surv[rng.integers(0, len(surv), len(kill))]
        tstar = (X[src] > lvl).argmax(1)
        for ts in np.unique(tstar):
            m = tstar == ts
            ki, si = kill[m], src[m]
            n = len(ki)
            w = Walk(n, L)
            for t in range(ts):
                tok = y[si, t]; tp = w.top()
                w.step(np.where(tok >= 2, 2, np.where(tok == tp, 0, 1)))
            for t in range(ts, L):
                p = w.law(); u = rng.random(n)
                w.step((u > p[:, 0]).astype(np.int64) + (u > p[:, 0] + p[:, 1]))
            nx = xi(w.y, t_from=ts + 1)
            nx[:, :ts + 1] = X[si, :ts + 1]
            y[ki] = w.y; X[ki] = nx
            cost.add_range(ts + 1, L, n)
        S = X.max(1)
        if cost.total > 50 * budget:
            break
    return math.exp(logp) * float((S > 1.0).mean()), cost.total


def ams_ce(model, lang, L, tau, budget, rng, f_ce=0.3, **kw):
    """AMS whose importance function is the CE-fitted log Q(prefix) - log P(prefix) (~ log P(event | prefix))."""
    _, spent, tw = _ce_fit(model, lang, L, tau, f_ce * budget, rng)

    def imp(yy):
        n = len(yy)
        w = Walk(n, L)
        out = np.zeros((n, L))
        acc = np.zeros(n)
        for t in range(L):
            out[:, t] = acc
            if t == L - 1:
                break
            p, _ = tw.probs(w); pl = w.law()
            tok = yy[:, t]; tp = w.top()
            a = np.where(tok >= 2, 2, np.where(tok == tp, 0, 1))
            acc = acc + np.log(p[np.arange(n), a]) - np.log(pl[np.arange(n), a])
            w.step(a)
        return 1 / (1 + np.exp(-out / 3))
    return ams(model, lang, L, tau, budget, rng, imp=imp, cost0=spent, **kw)


def ams_look(model, lang, L, tau, budget, rng, **kw):
    return ams(model, lang, L, tau, budget, rng, look=True, **kw)


# ----------------------------------------------------------------------------- CE-fitted twisted IS
class Twist:
    def __init__(self, nf, alpha=0.1):
        self.th = np.zeros((nf, 3))
        self.th[0] = [math.log(Q / 2), math.log(Q / 2), math.log(1 - Q)]  # = the law at d > 0
        self.alpha = alpha

    def probs(self, w):
        F = w.features()
        lg = F @ self.th
        lg[w.d == 0, 2] = -np.inf
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        return (1 - self.alpha) * p + self.alpha * w.law(), F

    def run(self, N, L, rng):
        w = Walk(N, L)
        llr = np.zeros(N)
        Fs, As = [], []
        for t in range(L - 1):  # y[L-1] never matters: sample it from the law
            p, F = self.probs(w)
            u = rng.random(N)
            a = (u > p[:, 0]).astype(np.int64) + (u > p[:, 0] + p[:, 1])
            pl = w.law()
            llr += np.log(pl[np.arange(N), a]) - np.log(p[np.arange(N), a])
            Fs.append(F); As.append(a)
            w.step(a)
        p = w.law(); u = rng.random(N)
        w.step((u > p[:, 0]).astype(np.int64) + (u > p[:, 0] + p[:, 1]))
        return w.y, llr, np.stack(Fs, 1), np.stack(As, 1), w

    def fit(self, Fs, As, wts, masks, iters=60, l2=1e-2):
        th = torch.tensor(self.th, requires_grad=True)
        F = torch.from_numpy(Fs.reshape(-1, Fs.shape[-1]))
        A = torch.from_numpy(As.reshape(-1))
        W = torch.from_numpy(np.repeat(wts / wts.sum(), Fs.shape[1]))
        M = torch.from_numpy(masks.reshape(-1, 3))
        opt = torch.optim.LBFGS([th], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            lg = (F @ th).masked_fill(~M, -1e30)
            ll = torch.log_softmax(lg, -1).gather(1, A[:, None])[:, 0]
            loss = -(W * ll).sum() + l2 * (th[1:] ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.th = th.detach().numpy()


def ce_is(model, lang, L, tau, budget, rng, rho=0.1, alpha=0.05, frac_final=0.5, n_stages=10, n_refine=2,
          return_twist=False, score=None, return_samples=False):
    """Cumulative cross-entropy method: each stage samples from the current twisted law, raises the level to the
    (1 - rho) quantile of the scores (capped at tau), and refits on the elites of ALL stages so far (weights =
    likelihood ratio x 1[score >= level]); after the level reaches tau, n_refine more stages at tau. The final
    estimate uses fresh samples only. Charged for every sample (CE stages too).
    Optional (added after the freeze, default unchanged): score(y) -> (S, Z) replaces (max_t p_illegal, 1) by a
    custom CE score S and integrand Z (estimate = mean LR x Z x 1[S > tau]); return_samples returns (y, LR)."""
    sc = score or (lambda yy: (phi_all(model, lang, yy)[:, :L].max(1), 1.0))
    nf = Walk(1, L).features().shape[1]
    tw = Twist(nf, alpha)
    n_stage = max(500, int(budget * (1 - frac_final) / n_stages))
    spent = 0.0
    keep = []  # (Fs, As, masks, llr, S)
    g, at_tau = -1.0, 0
    for it in range(n_stages):
        y, llr, Fs, As, w = tw.run(n_stage, L, rng)
        S = sc(y)[0]
        spent += n_stage
        dep, _, _ = stack_info(y)
        masks = np.ones((n_stage, L - 1, 3), bool); masks[:, :, 2] = dep[:, :L - 1] > 0
        keep.append((Fs, As, masks, llr, S))
        Sall = np.concatenate([k[4] for k in keep])
        g = max(g, min(tau, float(np.quantile(S, 1 - rho))))
        el = [k[4] >= g if g < tau else k[4] > tau for k in keep]
        if sum(e.sum() for e in el) < 5:
            continue
        lw = np.concatenate([k[3][e] for k, e in zip(keep, el)])
        wts = np.exp(lw - lw.max())
        tw.fit(np.concatenate([k[0][e] for k, e in zip(keep, el)]), np.concatenate([k[1][e] for k, e in zip(keep, el)]),
               wts, np.concatenate([k[2][e] for k, e in zip(keep, el)]))
        if g >= tau:
            at_tau += 1
            if at_tau > n_refine:
                break
    n_fin = max(500, int(budget - spent))
    y, llr, _, _, _ = tw.run(n_fin, L, rng)
    S, Z = sc(y)
    spent += n_fin
    est = float((np.exp(llr) * Z * (S > tau)).mean())
    if return_samples:
        return est, spent, tw, y, np.exp(llr)
    return (est, spent, tw) if return_twist else (est, spent)


# ----------------------------------------------------------------------------- top-risk enumeration
def replay(prefixes, L):
    """Walk positioned after each prefix (all the same length)."""
    P = np.array(prefixes, np.int64).reshape(len(prefixes), -1)
    w = Walk(len(P), L)
    for t in range(P.shape[1]):
        tok = P[:, t]; tp = w.top()
        w.step(np.where(tok >= 2, 2, np.where(tok == tp, 0, 1)))
    return w


def toprisk(model, lang, L, tau, budget, rng, f_ce=0.3, f_head=0.35, ce_kw=None):
    """(1) CE stages (f_ce of the budget) fit a twisted law Q ~ P(. | event). (2) Deductive head: best-first
    expansion of the prefix tree by log Q(prefix); every child is evaluated, a child whose running max exceeds
    tau is closed with its exact language mass P(prefix). (3) Sampled tail: fresh paths from Q; a path counts
    LR x 1[event] only if it leaves the expanded tree through an unclosed frontier node (paths through closed
    or exhausted nodes are already counted exactly). Unbiased; head deductive, tail sampling-based."""
    ce_kw = ce_kw or {}
    _, spent, tw = _ce_fit(model, lang, L, tau, f_ce * budget, rng, **ce_kw)
    cost = Cost(model, L)
    cost.total = spent
    lq2, l1q, lh = math.log(Q / 2), math.log(1 - Q), math.log(.5)
    root_phi = phi_at(model, lang, np.zeros((1, 0), np.int64), 0)[0]
    cost.add(0)
    if root_phi > tau:
        return 1.0, cost.total
    exact = 0.0
    expanded = set()
    heap = [(0.0, 0, (), 0.0, root_phi)]  # (-logQ, id, prefix, logP, runmax)
    cnt = 1
    head_end = cost.total + f_head * budget
    while heap and cost.total < head_end:
        batch = [heapq.heappop(heap) for _ in range(min(len(heap), 512))]
        bylen = {}
        for it in batch:
            bylen.setdefault(len(it[2]), []).append(it)
        for t, items in bylen.items():
            w = replay([it[2] for it in items], L)
            pq, _ = tw.probs(w)
            pl = w.law()
            kids, klp, klq, krm = [], [], [], []
            for j, (nq, _, pre, lp, rm) in enumerate(items):
                expanded.add(pre)
                top = int(w.top()[j]); d = int(w.d[j])
                acts = (0, 1, 2) if d > 0 else (0, 1)
                for a in acts:
                    tok = top if a == 0 else (1 - top if a == 1 else 2 + top)
                    kids.append(pre + (tok,)); klp.append(lp + math.log(pl[j, a])); klq.append(-nq + math.log(pq[j, a]))
                    krm.append(rm)
            if t + 1 > L - 1:
                continue
            Y = np.array(kids, np.int64)
            ph = phi_at(model, lang, Y, t + 1)
            cost.add(t + 1, len(kids))
            for k, lp, lq, rm, p in zip(kids, klp, klq, krm, ph):
                r = max(rm, p)
                if r > tau:
                    exact += math.exp(lp)
                elif t + 1 < L - 1:
                    cnt += 1
                    heapq.heappush(heap, (-lq, cnt, k, lp, r))
                else:
                    expanded.add(k)  # exhausted leaf: no event, resolved
    n_fin = max(500, int(budget - cost.total))
    y, llr, _, _, _ = tw.run(n_fin, L, rng)
    S = phi_all(model, lang, y)
    cost.total += n_fin
    unc = np.zeros(n_fin, bool)
    for i in range(n_fin):
        row = tuple(y[i, :L - 1].tolist())
        t = 0
        while t <= L - 1 and row[:t] in expanded:
            t += 1
        if t > L - 1:
            continue  # every prefix resolved in the head: exhausted path without event
        # row[:t] is an evaluated child of an expanded node (t >= 1): closed iff its running max > tau
        unc[i] = t == 0 or S[i, :t + 1].max() <= tau
    S = S[:, :L].max(1)
    return exact + float((np.exp(llr) * (S > tau) * unc).mean()), cost.total


def _ce_fit(model, lang, L, tau, budget, rng, **kw):
    """CE stages only (no final estimate): returns (None, cost, twist)."""
    nf = Walk(1, L).features().shape[1]
    est, spent, tw = ce_is(model, lang, L, tau, budget, rng, frac_final=0.0, return_twist=True, **kw)
    return None, spent, tw


METHODS = dict(mc=plain_mc, ams=ams, amsl=ams_look, amsce=ams_ce, ce=ce_is, top=toprisk)
