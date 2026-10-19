"""T445 (S2): estimators of Q = E[(1/L) sum_t p_illegal(x, t)] for dyck2 at L = 32.

Each estimator: est(model, lang, B, rng) -> (estimate, charged_cost_in_FPE).
Charging: a forward over an input of length n costs model.flops(n) / model.flops(L) FPE.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mini_lm"))
from langs import model_input  # noqa: E402

L = 32
Q_OPEN = 0.5


@torch.no_grad()
def p_illegal(model, lang, y, chunk=8192):
    """(N, T) probability mass on illegal next tokens at every position of y (N, T)."""
    model.eval()
    legal = lang.annotate(y)["legal"]
    x = model_input(y, lang.BOS)
    out = []
    for i in range(0, len(x), chunk):
        p = torch.softmax(model(torch.from_numpy(x[i:i + chunk])).double(), -1).numpy()
        out.append((p * ~legal[i:i + chunk]).sum(-1))
    return np.concatenate(out)


def cost(model, n_tokens, count=1):
    return count * model.flops(n_tokens) / model.flops(L)


# ---------------- E0: plain Monte Carlo ----------------
def e0_plain(model, lang, B, rng):
    n = int(B)
    y = lang.sample(n, rng, L=L)
    return float(p_illegal(model, lang, y).mean()), float(n)


# ---------------- E1: exact head + MC tail ----------------
_head_cache = {}


def head_t0(model, lang, B):
    t0 = 1
    while t0 + 1 <= L and cost(model, t0 + 1, lang.count(t0)) <= B / 2:
        t0 += 1
    return t0


def e1_head_tail(model, lang, B, rng):
    """Positions t < t0 summed exactly over every prefix of t0 - 1 tokens (input length t0);
    positions t >= t0 estimated from MC sequences with the remaining budget."""
    t0 = head_t0(model, lang, B)
    yh, wh = lang.enumerate_small(T=t0 - 1) if t0 > 1 else (np.zeros((1, 0), np.int64), np.ones(1))
    c_head = cost(model, t0, len(wh))
    if t0 > 1:
        yh = np.concatenate([yh, np.zeros((len(yh), 1), np.int64)], 1)  # dummy last target (position t0-1 is scored
        pi = p_illegal_prefix(model, lang, yh)                          # from the prefix only)
        head = float((wh[:, None] * pi).sum())
    else:
        pi = p_illegal_prefix(model, lang, np.zeros((1, 1), np.int64))
        head = float(pi.sum())
    n = max(int(B - c_head), 1)
    y = lang.sample(n, rng, L=L)
    tail = float(p_illegal(model, lang, y)[:, t0:].sum(1).mean())
    return (head + tail) / L, c_head + n


@torch.no_grad()
def p_illegal_prefix(model, lang, y):
    """p_illegal at positions 0..T-1 where the last column of y is a placeholder: legality at position t depends only
    on y[:, :t]; the placeholder token is never used as an input (inputs are BOS + y[:, :-1])."""
    T = y.shape[1]
    yy = y.copy()
    # make the placeholder legal so annotate accepts it: an opener is always legal
    yy[:, -1] = 0
    return p_illegal(model, lang, yy)


# ---------------- depth-chain tables (model-free, free of charge) ----------------
class DepthChain:
    """Depth process of dyck2: from 0 up w.p. 1; from d > 0 up w.p. q, down w.p. 1 - q."""

    def __init__(self, Lmax=L, q=Q_OPEN):
        self.L, self.q = Lmax, q
        D = Lmax + 2
        up = np.full(D, q); up[0] = 1.0
        self.up = up
        A = np.zeros((Lmax + 1, D)); A[0, 0] = 1
        for s in range(Lmax):
            A[s + 1, 1:] += A[s, :-1] * up[:-1]
            A[s + 1, :-1] += A[s, 1:] * (1 - up[1:])
        self.A = A  # A[s, e] = P(depth e after s tokens)
        # ballot table: number of paths with r steps from height h back to 0 staying >= 0
        N2 = np.zeros((Lmax + 1, D)); N2[0, 0] = 1
        for r in range(1, Lmax + 1):
            N2[r, :-1] += N2[r - 1, 1:]          # up
            N2[r, 1:] += N2[r - 1, :-1]          # down
        self.N2 = N2
        # backward hitting tables H[j][e_target][s, e] = P(depth e_target at time j | depth e at time s)
        self._H = {}

    def H(self, j, et):
        key = (j, et)
        if key not in self._H:
            D = self.L + 2
            h = np.zeros((j + 1, D)); h[j, et] = 1
            for s in range(j - 1, -1, -1):
                h[s, :-1] += self.up[:-1] * h[s + 1, 1:]
                h[s, 1:] += (1 - self.up[1:]) * h[s + 1, :-1]
            self._H[key] = h
        return self._H[key]

    def exc(self, n):
        """P(excursion of n steps from depth d >= 1 returning to d without going below d)."""
        if n % 2:
            return 0.0
        return self.N2[n, 0] * (self.q * (1 - self.q)) ** (n // 2)

    def items(self):
        """All (t, d, j, P) with P = P(depth d_t = d and matching opener at j) at position t (d = 0: j = -1)."""
        out = []
        for t in range(L):
            out.append((t, 0, -1, self.A[t, 0]))
            for j in range(t):
                for d in range(1, j + 2):
                    p = self.A[j, d - 1] * self.up[d - 1] * self.exc(t - 1 - j)
                    if p > 0:
                        out.append((t, d, j, p))
        return np.array(out)

    def sample_paths(self, t, d, j, rng):
        """Token prefixes (len(t),) arrays -> list of y-prefixes of length t with the exact conditional law."""
        n = len(t)
        Tm = int(t.max()) if n else 0
        dep = np.zeros((n, Tm + 1), np.int64)
        moves = np.zeros((n, Tm), np.int64)  # +1 open, -1 close
        for i in range(n):  # loop over items; each path at most 32 steps
            ti, di, ji = int(t[i]), int(d[i]), int(j[i])
            e = 0
            if di == 0:
                h = self.H(ti, 0)
                seg1 = ti
            else:
                h = self.H(ji, di - 1)
                seg1 = ji
            for s in range(seg1):
                pu = self.up[e] * h[s + 1, e + 1] / h[s, e]
                m = 1 if rng.random() < pu else -1
                moves[i, s] = m; e += m
            if di > 0:
                moves[i, ji] = 1; e += 1
                r = ti - 1 - ji
                hh = 0  # relative height of the excursion
                for s in range(r):
                    rem = r - s - 1
                    nu = self.N2[rem, hh + 1]
                    nd = self.N2[rem, hh - 1] if hh > 0 else 0.0
                    m = 1 if rng.random() * (nu + nd) < nu else -1
                    moves[i, ji + 1 + s] = m; hh += m
        return moves


def moves_to_tokens(moves, lens, rng):
    """Depth moves -> dyck2 tokens: openers get uniform types, closers match the stack."""
    n, T = moves.shape
    y = np.zeros((n, T), np.int64)
    for i in range(n):
        st = []
        for s in range(int(lens[i])):
            if moves[i, s] == 1:
                ty = int(rng.random() < 0.5)
                st.append(ty); y[i, s] = ty
            else:
                y[i, s] = 2 + st.pop()
    return y


# ---------------- E2: tilted importance sampling ----------------
def sample_tilted(lang, n, qp, rng):
    """dyck2 sequences under open prob qp at depth > 0; returns y and per-position prefix log-weights (N, L)."""
    y = np.empty((n, L), np.int64)
    lw = np.zeros((n, L))
    stack = np.zeros((n, L + 1), np.int64)
    d = np.zeros(n, np.int64)
    idx = np.arange(n)
    acc = np.zeros(n)
    q = Q_OPEN
    for t in range(L):
        lw[:, t] = acc  # weight of the prefix y[:, :t] (position t is predicted from it)
        u = rng.random(n)
        typ = (rng.random(n) < 0.5).astype(np.int64)
        close = (d > 0) & (u >= qp)
        top = stack[idx, np.maximum(d - 1, 0)]
        y[:, t] = np.where(close, 2 + top, typ)
        op = ~close
        stack[idx[op], d[op]] = typ[op]
        inner = d > 0
        acc = acc + np.where(inner & op, np.log(q / qp), 0.0) + np.where(inner & close, np.log((1 - q) / (1 - qp)), 0.0)
        d = d + np.where(close, -1, 1)
    return y, lw


def e2_tilted(model, lang, B, rng, grid=(0.5, 0.55, 0.6, 0.65, 0.7), pilot_frac=0.02):
    spent = 0.0
    best, best_m2 = grid[0], np.inf
    npil = max(int(pilot_frac * B), 2)
    for qp in grid:
        y, lw = sample_tilted(lang, npil, qp, rng)
        z = (np.exp(lw) * p_illegal(model, lang, y)).mean(1)
        spent += npil
        m2 = float((z ** 2).mean())
        if m2 < best_m2:
            best, best_m2 = qp, m2
    n = max(int(B - spent), 1)
    y, lw = sample_tilted(lang, n, best, rng)
    est = float((np.exp(lw) * p_illegal(model, lang, y)).mean())
    return est, spent + n


# ---------------- E3: feature importance sampling over positions ----------------
KB = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32)]
_chain = None


def chain():
    global _chain
    if _chain is None:
        _chain = DepthChain()
        it = _chain.items()
        cell = np.zeros(len(it), np.int64)
        k = it[:, 0] - it[:, 2]
        for c, (a, b) in enumerate(KB):
            cell[(it[:, 1] >= 1) & (k >= a) & (k <= b)] = c + 1
        _chain.it, _chain.cell = it, cell
        _chain.Pcell = np.array([it[cell == c, 3].sum() for c in range(len(KB) + 1)])  # sums to L
    return _chain


def _items_pill(model, lang, ch, ids, rng):
    """p_illegal at the item position (one forward over the prefix, input length t + 1) and the charged cost."""
    it = ch.it[ids]
    t, d, j = it[:, 0].astype(int), it[:, 1].astype(int), it[:, 2].astype(int)
    moves = ch.sample_paths(t, d, j, rng)
    y = moves_to_tokens(moves, t, rng)
    out = np.zeros(len(ids)); c = 0.0
    for tt in np.unique(t):
        m = t == tt
        yy = np.concatenate([y[m][:, :tt], np.zeros((m.sum(), 1), np.int64)], 1)
        out[m] = p_illegal_prefix(model, lang, yy)[:, tt]
        c += cost(model, tt + 1, int(m.sum()))
    return out, c


def e3_feature_is(model, lang, B, rng, pilot_frac=0.2, floor=0.01):
    ch = chain()
    P = ch.it[:, 3]
    mean_cost = float((P * np.array([model.flops(int(t) + 1) for t in ch.it[:, 0]])).sum() / P.sum() / model.flops(L))
    # pilot: items from the law
    n_p = max(int(pilot_frac * B / mean_cost), 12)
    ids = rng.choice(len(P), n_p, p=P / P.sum())
    pv, c_p = _items_pill(model, lang, ch, ids, rng)
    ncell = len(ch.Pcell)
    m2 = np.array([(pv[ch.cell[ids] == c] ** 2).mean() if (ch.cell[ids] == c).any() else 0.0 for c in range(ncell)])
    share = ch.Pcell / ch.Pcell.sum()
    pi = share * np.sqrt(m2)
    pi = pi / pi.sum() if pi.sum() > 0 else share.copy()
    pi = np.maximum(pi, floor * share); pi /= pi.sum()
    # main: draw cells by pi, items within a cell by P
    budget = B - c_p
    cell_cost = np.array([(P[ch.cell == c] * np.array([model.flops(int(t) + 1) for t in ch.it[ch.cell == c, 0]])).sum()
                          / P[ch.cell == c].sum() / model.flops(L) for c in range(ncell)])
    n_m = max(int(budget / float((pi * cell_cost).sum())), 1)
    cells = rng.choice(ncell, n_m, p=pi)
    est_sum, c_m = 0.0, 0.0
    for c in range(ncell):
        k = int((cells == c).sum())
        if k == 0:
            continue
        pc = P[ch.cell == c]
        ids_c = np.nonzero(ch.cell == c)[0][rng.choice(len(pc), k, p=pc / pc.sum())]
        v, cc = _items_pill(model, lang, ch, ids_c, rng)
        est_sum += float((v * (ch.Pcell[c] / pi[c])).sum())  # weight = P(cell) / pi(cell); P(cell) sums to L
        c_m += cc
    return est_sum / n_m / L, c_p + c_m


# ---------------- E5: exact head (<= B/4) + tilted-IS tail (amendment before panel scoring) ----------------
def e5_head_tilted(model, lang, B, rng, grid=(0.5, 0.55, 0.6, 0.65, 0.7), pilot_frac=0.02):
    t0 = head_t0(model, lang, B / 2)  # head cost <= B/4
    if t0 > 1:
        yh, wh = lang.enumerate_small(T=t0 - 1)
        yh = np.concatenate([yh, np.zeros((len(yh), 1), np.int64)], 1)
    else:
        yh, wh = np.zeros((1, 1), np.int64), np.ones(1)
    head = float((wh[:, None] * p_illegal_prefix(model, lang, yh)).sum())
    spent = cost(model, t0, len(wh))
    best, best_m2 = grid[0], np.inf
    npil = max(int(pilot_frac * B), 2)
    for qp in grid:
        y, lw = sample_tilted(lang, npil, qp, rng)
        z = (np.exp(lw) * p_illegal(model, lang, y))[:, t0:].sum(1)
        spent += npil
        m2 = float((z ** 2).mean())
        if m2 < best_m2:
            best, best_m2 = qp, m2
    n = max(int(B - spent), 1)
    y, lw = sample_tilted(lang, n, best, rng)
    tail = float((np.exp(lw) * p_illegal(model, lang, y))[:, t0:].sum(1).mean())
    return (head + tail) / L, spent + n


ESTIMATORS = {"E0": e0_plain, "E1": e1_head_tail, "E2": e2_tilted, "E3": e3_feature_is, "E5": e5_head_tilted}
