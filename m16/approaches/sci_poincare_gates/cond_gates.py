"""T376: T3's reduced system with the low inputs made consistent with the symbol's cylinder.
Low slot t (rank > k) gets x_t = Phi^-1(ubar * Phi(c)), where ubar = E[u_t | cell], u_t = Phi(x_t)/Phi(c), and the
cell is the symbol S (gate itinerary of set C) plus the top-k skeleton. Conditional means are estimated on an
independent fit sample (cross-fitted) with a fallback hierarchy (fewer than NMIN fit points -> next level):
  L0 (S, ordered top-k positions) -> L1 (S, unordered top-k position set) -> L2 (S) -> L3 E27 V1.
Usage: python cond_gates.py [k] [model names...]. Writes results/cond_k<k>_<model>.json."""
import sys, json, os
import numpy as np
from scipy.special import ndtr, ndtri
from models import all_models, clocks_data
from ceiling_gates import run

NFIT, NTEST, CH, NMIN = 4_000_000, 600_000, 250_000, 20


def itin(B, cols):
    return (B[:, :, cols].reshape(len(B), -1) @ (1 << np.arange(10 * int(np.sum(cols)))).astype(np.int64)).astype(np.int64)


def skel(o, k):
    """L0: ordered top-k positions (base 10); L1: unordered set (bitmask)."""
    p = o[:, :k]
    return (p * 10 ** np.arange(k)).sum(1), (1 << p).sum(1)


def uvals(x, o, k):
    c = np.take_along_axis(x, o[:, k-1:k], 1)
    low = np.ones_like(x, bool); np.put_along_axis(low, o[:, :k], False, 1)
    return np.where(low, ndtr(x) / ndtr(c), 0.0), low, c[:, 0]


class Table:
    """Per-level sums of u and counts per slot. Symbols are mapped to dense ids so keys fit in int64."""
    def __init__(self): self.S = []
    def add(self, S): self.S.append(S)
    def keys(self, S, s0, s1):
        i = np.clip(np.searchsorted(self.uS, S), 0, len(self.uS) - 1); sid = np.where(self.uS[i] == S, i, -1)
        return sid, (sid << 17) + s0, (sid << 17) + s1
    def build(self, s0, s1, u, low):
        S = np.concatenate(self.S); del self.S; self.uS = np.unique(S); self.lv = []
        sid, k0, k1 = self.keys(S, s0, s1)
        for K in (k0, k1, sid):
            uk, inv = np.unique(K, return_inverse=True)
            n = np.stack([np.bincount(inv, low[:, t], len(uk)).astype(np.int32) for t in range(10)], 1)
            keep = n.max(1) >= NMIN; remap = np.cumsum(keep) - 1; m = keep[inv]; iv = remap[inv[m]]   # prune sparse cells
            su = np.stack([np.bincount(iv, u[m, t], keep.sum()).astype(np.float32) for t in range(10)], 1)
            self.lv.append((uk[keep], su, n[keep])); del inv, m, iv
    def lookup(self, S, s0, s1, low, k):
        out = np.full(low.shape, np.nan); level = np.full(low.shape, 3)
        sid, k0, k1 = self.keys(S, s0, s1)
        for L, K in enumerate((k0, k1, sid)):
            uk, su, n = self.lv[L]; pos = np.clip(np.searchsorted(uk, K), 0, len(uk) - 1)
            hit = (uk[pos] == K) & (sid >= 0)
            nn = np.where(hit[:, None], n[pos], 0); ok = low & np.isnan(out) & (nn >= NMIN)
            out[ok] = (su[pos] / np.maximum(nn, 1))[ok]; level[ok] = L
        return out, level


def v1_u(n_, rows):
    j = np.arange(1, n_ + 1); return np.tile((n_ + 1 - j) / (n_ + 1), (rows, 1))


def main(k, names):
    ms = all_models(); os.makedirs("results", exist_ok=True)
    for name in names:
        M = ms[name]; d5 = clocks_data(M, 5); rest = [j for j in range(16) if j not in d5]
        sets = {"D3": d5[:3], "D5": d5, "R3": sorted(np.random.default_rng(11).choice(rest, 3, replace=False).tolist())}
        fm = {s: np.isin(np.arange(16), v) for s, v in sets.items()}
        # one fit sample shared by all tables; sk = symbol-free control (same estimator, cell = skeleton only)
        tabs = {s: Table() for s in sets}; sk = Table(); rf = np.random.default_rng(101); U, LOW, S0, S1 = [], [], [], []
        for _ in range(NFIT // CH):
            x = rf.standard_normal((CH, 10)); o = np.argsort(-x, 1); _, B = run(M, x)
            u, low, _ = uvals(x, o, k); s0, s1 = skel(o, k)
            U.append(u.astype(np.float32)); LOW.append(low); S0.append(s0); S1.append(s1)
            for s in sets: tabs[s].add(itin(B, fm[s]))
            sk.add(np.zeros(CH, np.int64))
        U, LOW, S0, S1 = (np.concatenate(a) for a in (U, LOW, S0, S1))
        for t in list(tabs.values()) + [sk]: t.build(S0, S1, U, LOW)
        del U, LOW, S0, S1
        rng = np.random.default_rng(7); rng_x = np.random.default_rng(13)
        diff, flips, lev, viol = {}, {}, {}, {}
        for _ in range(NTEST // 50_000):
            x = rng.standard_normal((50_000, 10)); o = np.argsort(-x, 1); tgt = o[:, 1]; n = len(x)
            lg, B = run(M, x); ok = lg.argmax(1) == tgt
            u, low, c = uvals(x, o, k); s0, s1 = skel(o, k)
            xs = x.copy(); uu = rng_x.random((n, 10 - k)) * ndtr(c)[:, None]
            np.put_along_axis(xs, o[:, k:], ndtri(np.maximum(uu, 1e-300)), 1); _, Bs = run(M, xs)
            def build(tab, S):
                ub, L = tab.lookup(S, s0, s1, low, k)
                fb = low & np.isnan(ub); v1 = np.zeros_like(ub); np.put_along_axis(v1, o[:, k:], v1_u(10 - k, n), 1)
                ub = np.where(fb, v1, ub); xr = np.where(low, ndtri(np.clip(ub, 1e-12, 1) * ndtr(c)[:, None]), x)
                return xr, L
            variants = {}
            for s in sets:
                xr, L = build(tabs[s], itin(B, fm[s])); variants[f"{s}"] = (xr, fm[s], B, L, s)
            xr, L = build(tabs["D3"], itin(Bs, fm["D3"])); variants["X3"] = (xr, fm["D3"], Bs, L, "D3x")
            xr, L = build(sk, np.zeros(n, np.int64)); variants["skel"] = (xr, None, None, L, None)
            variants["D3u"] = (variants["D3"][0], None, None, variants["D3"][3], None)   # conditioned, not forced
            # T3 diagnostic: cylinder violation of E27/T3's V1 frozen lows under the D3 symbol
            xv = x.copy(); np.put_along_axis(xv, o[:, k:], ndtri(v1_u(10 - k, n) * ndtr(c)[:, None]), 1); _, Bv = run(M, xv)
            viol["T3_V1"] = viol.get("T3_V1", 0) + int((itin(Bv, fm["D3"]) != itin(B, fm["D3"])).sum())
            for key, (xr, f, Bm, L, s) in variants.items():
                lr, Br = run(M, xr, f, Bm); okr = lr.argmax(1) == tgt
                diff[key] = diff.get(key, 0) + int(okr.sum() - ok.sum()); flips[key] = flips.get(key, 0) + int((okr != ok).sum())
                lev[key] = lev.get(key, np.zeros(4)) + np.bincount(L[low], minlength=4)
                if s in sets:   # cylinder violation: does the unforced network on xr reproduce S?
                    _, Bu = run(M, xr); viol[key] = viol.get(key, 0) + int((itin(Bu, fm[s]) != itin(B, fm[s])).sum())
        out = {"model": name, "k": k, "N": NTEST, "NFIT": NFIT, "NMIN": NMIN, "sets": sets,
               "bias_pp": {kk: 100 * v / NTEST for kk, v in diff.items()},
               "se_pp": {kk: 100 * np.sqrt(flips[kk]) / NTEST for kk in diff},
               "level_frac": {kk: (v / v.sum()).round(4).tolist() for kk, v in lev.items()},
               "cyl_violation": {kk: v / NTEST for kk, v in viol.items()}}
        json.dump(out, open(f"results/cond_k{k}_{name}.json", "w"), indent=1)
        b = out["bias_pp"]
        print(f"{name:10s} k{k} | " + " ".join(f"{s} {b[s]:+.3f}" for s in b) + " | L0 frac D3 "
              f"{out['level_frac']['D3'][0]:.2f} viol D3 {out['cyl_violation']['D3']:.3f}", flush=True)


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(k, sys.argv[2:] or list(all_models()))
