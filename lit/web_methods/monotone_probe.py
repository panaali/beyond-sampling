"""Monotonicity probe for preintegration along tie segments (Gilbert-Kuo-Sloan 2022 condition).
Classic preintegration theory needs the integrand to switch once along the preintegrated line. For each
random x we take the sci_tao-style segment through the (2nd,3rd) and (1st,2nd) tie faces, direction
(e_a-e_b)/sqrt2, restricted so a,b stay that pair, evaluate correctness on a 256-point grid and count
wrong runs. 'Non-monotone' = >=2 wrong runs, or a wrong run not touching the tie point.
Reports, per (H,T): share of segments with any error, and the share of (phi-weighted) error mass carried by
non-monotone segments. Grid resolution misses runs thinner than range/256 (undercounts).
Segments are clipped to the sci_tao region of the face (nearest adjacent tie) around the tie point.
Usage: OMP_NUM_THREADS=1 python lit/web_methods/monotone_probe.py [n_lines] [H] [T ...]"""
import sys, numpy as np
from alg_zoo import zoo_2nd_argmax, example_2nd_argmax

def load(H, T):
    m = example_2nd_argmax() if (H, T) == (16, 10) else zoo_2nd_argmax(H, T)
    f = lambda w: w.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)

def correct(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for s in range(x.shape[1]):
        h = np.maximum(h @ Wh.T + x[:, s:s+1] * Wi, 0)
    return (h @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]

def probe(model, T, n, rng, pair, G=256):
    stats = dict(lines=0, err_lines=0, nm_lines=0, mass=0.0, nm_mass=0.0)
    for _ in range(n // 200):
        X = rng.standard_normal((200, T))
        for x in X:
            o = np.argsort(-x); v = x[o]
            a, b = (o[1], o[2]) if pair == 23 else (o[0], o[1])
            lo_c = (v[3] if T > 3 else -np.inf) if pair == 23 else v[2]; hi_c = v[0] if pair == 23 else np.inf
            r = 1 / np.sqrt(2)  # x_a + s r, x_b - s r must stay in (lo_c, hi_c)
            smax = min((hi_c - x[a]) / r, (x[b] - lo_c) / r); smin = max((lo_c - x[a]) / r, (x[b] - hi_c) / r)
            smax = min(smax, 6.0); smin = max(smin, -6.0)
            s = np.linspace(smin, smax, G)
            L = np.repeat(x[None], G, 0); L[:, a] += s * r; L[:, b] -= s * r
            # keep the sci_tao region of this face: R23={v2-v3 < v1-v2}, R12={v1-v2 <= v2-v3}, and the
            # connected piece around the tie point, so bands of neighbouring tie faces are excluded
            V = -np.sort(-L, 1); g12 = V[:, 0] - V[:, 1]; g23 = V[:, 1] - V[:, 2]
            ok = (g23 < g12) if pair == 23 else (g12 <= g23)
            s0 = (x[b] - x[a]) / (2 * r); i0 = min(max(np.searchsorted(s, s0), 0), G - 1)
            lab = np.cumsum(np.r_[1, np.diff(ok.astype(int)) != 0]); keep = ok & (lab == lab[i0])
            if not keep.any():
                continue
            s, L = s[keep], L[keep]
            w = np.exp(-0.5 * (L**2).sum(1)); wrong = ~correct(model, L)
            stats['lines'] += 1
            if not wrong.any():
                continue
            stats['err_lines'] += 1
            runs = np.flatnonzero(np.diff(np.r_[0, wrong.astype(int), 0]) == 1)
            ends = np.flatnonzero(np.diff(np.r_[0, wrong.astype(int), 0]) == -1)
            i0 = np.searchsorted(s, s0)
            touch = [(st <= i0 + 1) and (en >= i0 - 1) for st, en in zip(runs, ends)]
            m = (w * wrong).sum() * ((smax - smin) / (G - 1))
            stats['mass'] += m
            if len(runs) >= 2 or not all(touch):
                stats['nm_lines'] += 1; stats['nm_mass'] += m
    return stats

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    Ts = [int(t) for t in sys.argv[3:]] or [4, 5, 6, 8, 10]
    for T in Ts:
        model = load(H, T)
        for pair in (23, 12):
            st = probe(model, T, n, np.random.default_rng(T * 100 + pair), pair)
            print(f"H={H} T={T} pair={pair}: lines with error {st['err_lines']}/{st['lines']}; "
                  f"non-monotone {st['nm_lines']}/{max(st['err_lines'],1)} lines, "
                  f"carrying {st['nm_mass']/max(st['mass'],1e-300):.2f} of error mass", flush=True)
