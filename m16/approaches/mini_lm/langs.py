"""Mini languages with exactly known next-token laws (T433).

Each language object provides
  sample(N, rng, **kw)      -> int64 targets y (N, L)  (model input is [BOS] + y[:, :-1])
  enumerate_small()         -> (y, w): every sequence of the small horizon with its exact probability
  annotate(y)               -> dict of per-position arrays (legal mask, true next-token law, scored mask, ...)
Positions are indexed by the target index t: position t predicts y[:, t] from BOS, y[:, :t].
"""
import numpy as np


def model_input(y, bos):
    x = np.empty_like(y)
    x[:, 0] = bos
    x[:, 1:] = y[:, :-1]
    return x


class Dyck2:
    """Dyck prefixes with 2 bracket types. ids: ( [ ) ] = 0 1 2 3, BOS = 4."""
    name = "dyck2"
    V_out, V_in, BOS = 4, 5, 4

    def __init__(self, L=32, q=0.5, T_small=15):
        self.L, self.q, self.T_small = L, q, T_small

    def sample(self, N, rng, L=None):
        L = L or self.L
        y = np.empty((N, L), np.int64)
        stack = np.zeros((N, L), np.int64)
        d = np.zeros(N, np.int64)
        idx = np.arange(N)
        for t in range(L):
            u = rng.random(N)
            typ = (rng.random(N) < 0.5).astype(np.int64)
            close = (d > 0) & (u >= self.q)
            top = stack[idx, np.maximum(d - 1, 0)]
            y[:, t] = np.where(close, 2 + top, typ)
            op = ~close
            stack[idx[op], d[op]] = typ[op]
            d = d + np.where(close, -1, 1)
        return y

    def count(self, T):
        c = {0: 1}
        for _ in range(T):
            n = {}
            for d, v in c.items():
                n[d + 1] = n.get(d + 1, 0) + 2 * v
                if d > 0:
                    n[d - 1] = n.get(d - 1, 0) + v
            c = n
        return sum(c.values())

    def enumerate_small(self, T=None, as_int8=False):
        """All length-T sequences with positive probability, exact float64 probabilities."""
        T = T or self.T_small
        y = np.zeros((1, 0), np.int8)
        st = np.zeros((1, T), np.int8)
        d = np.zeros(1, np.int64)
        lp = np.zeros(1)
        lq2, l1q, lh = np.log(self.q / 2), np.log(1 - self.q), np.log(0.5)
        for t in range(T):
            M = len(d)
            parts = []
            for typ in (0, 1):  # openers
                yy = np.concatenate([y, np.full((M, 1), typ, np.int8)], 1)
                ss = st.copy(); ss[np.arange(M), d] = typ
                parts.append((yy, ss, d + 1, lp + np.where(d > 0, lq2, lh)))
            m = d > 0
            if m.any():
                top = st[np.nonzero(m)[0], d[m] - 1]
                yy = np.concatenate([y[m], (2 + top)[:, None].astype(np.int8)], 1)
                parts.append((yy, st[m], d[m] - 1, lp[m] + l1q))
            y = np.concatenate([p[0] for p in parts]); st = np.concatenate([p[1] for p in parts])
            d = np.concatenate([p[2] for p in parts]); lp = np.concatenate([p[3] for p in parts])
        return (y if as_int8 else y.astype(np.int64)), np.exp(lp)

    def enumerate_chunks(self, T=None, chunk=200000):
        y, w = self.enumerate_small(T, as_int8=True)
        for i in range(0, len(y), chunk):
            yield y[i:i + chunk].astype(np.int64), w[i:i + chunk]

    def annotate(self, y):
        N, L = y.shape
        q = self.q
        legal = np.zeros((N, L, 4), bool)
        law = np.zeros((N, L, 4))
        depth = np.zeros((N, L), np.int64)
        correct = np.full((N, L), -1, np.int64)  # correct closer id when d > 0
        stack = np.zeros((N, L + 1), np.int64)
        d = np.zeros(N, np.int64)
        idx = np.arange(N)
        for t in range(L):
            depth[:, t] = d
            legal[:, t, :2] = True
            pos = d > 0
            top = stack[idx, np.maximum(d - 1, 0)]
            cc = 2 + top
            correct[pos, t] = cc[pos]
            legal[idx[pos], t, cc[pos]] = True
            law[:, t, :2] = np.where(pos, q / 2, 0.5)[:, None]
            law[idx[pos], t, cc[pos]] = 1 - q
            tok = y[:, t]
            op = tok < 2
            stack[idx[op], d[op]] = tok[op]
            d = d + np.where(op, 1, -1)
            if (d < 0).any():
                raise ValueError("illegal sequence")
        return dict(legal=legal, law=law, depth=depth, correct=correct,
                    scored=np.ones((N, L), bool))


class ModSum:
    """d_1..d_n = s, n ~ U{1..N}, s = sum mod m. ids: digits 0..m-1, '=' = m, BOS = m+1."""
    name = "modsum"

    def __init__(self, m=5, N=20, n_small=10):
        self.m, self.N, self.n_small = m, N, n_small
        self.EQ, self.BOS = m, m + 1
        self.V_out, self.V_in = m + 1, m + 2
        self.L = N + 2

    def _build(self, digits, n):
        Ns = len(n)
        y = np.full((Ns, self.L), self.EQ, np.int64)
        cols = np.arange(self.N)[None, :]
        live = cols < n[:, None]
        y[:, :self.N] = np.where(live, digits, self.EQ)
        s = (np.where(live, digits, 0).sum(1)) % self.m
        y[np.arange(Ns), n + 1] = s
        return y

    def sample(self, N, rng, n=None):
        if n is None:
            nn = rng.integers(1, self.N + 1, N)
        else:
            nn = np.full(N, n, np.int64)
        dig = rng.integers(0, self.m, (N, self.N))
        return self._build(dig, nn)

    def enumerate_chunks(self, n_max=None, chunk=200000):
        """Streams (y, w) over every digit string with n <= n_max (n uniform on 1..n_max)."""
        n_max = n_max or self.n_small
        for n in range(1, n_max + 1):
            K = self.m ** n
            for c0 in range(0, K, chunk):
                code = np.arange(c0, min(K, c0 + chunk))
                dig = np.zeros((len(code), self.N), np.int64)
                for i in range(n):
                    dig[:, i] = (code // self.m ** i) % self.m
                yield self._build(dig, np.full(len(code), n, np.int64)), np.full(len(code), 1.0 / (n_max * K))

    def enumerate_small(self, n_max=None, n_only=None):
        """All digit strings with n <= n_max (law conditioned on n <= n_max: n uniform); or n == n_only."""
        n_max = n_max or self.n_small
        ns = [n_only] if n_only else range(1, n_max + 1)
        ys, ws = [], []
        for n in ns:
            K = self.m ** n
            code = np.arange(K)
            dig = np.zeros((K, self.N), np.int64)
            for i in range(n):
                dig[:, i] = (code // self.m ** i) % self.m
            ys.append(self._build(dig, np.full(K, n, np.int64)))
            ws.append(np.full(K, 1.0 / (len(ns) * K)))
        return np.concatenate(ys), np.concatenate(ws)

    def annotate(self, y):
        Nn, L = y.shape
        m, EQ = self.m, self.EQ
        n = (y[:, :self.N] == EQ).argmax(1)
        n = np.where((y[:, :self.N] != EQ).all(1), self.N, n)
        legal = np.zeros((Nn, L, m + 1), bool)
        law = np.zeros((Nn, L, m + 1))
        t = np.arange(L)[None, :]
        for k in range(L):
            if k < self.N:
                h = 1.0 / (self.N + 1 - k)
                # after k digits (k <= n): '=' with hazard h, else uniform digit; only if k <= n
                pre = k <= n
                legal[pre, k, :m] = True
                legal[pre, k, EQ] = True
                law[pre, k, :m] = (1 - h) / m
                law[pre, k, EQ] = h
            else:
                pre = k <= n
                legal[pre, k, EQ] = True
                law[pre, k, EQ] = 1.0
        ans = n + 1
        s = y[np.arange(Nn), ans]
        legal[np.arange(Nn), ans] = False
        legal[np.arange(Nn), ans, s] = True
        law[np.arange(Nn), ans] = 0
        law[np.arange(Nn), ans, s] = 1
        scored = t <= ans[:, None]  # training loss positions
        return dict(legal=legal, law=law, n=n, ans_pos=ans, answer=s, scored=scored)


LANGS = {"dyck2": Dyck2, "modsum": ModSum}
