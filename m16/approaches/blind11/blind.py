"""T293 BLIND-11 (pre-registered in README). est_blind(model) -> (contribution, cost_flops), plus the DEV2/LB3 test with a sampled
diagnostic reference. usage (repo root): python m16/approaches/blind11/blind.py DEV2_WEIGHTS.npz"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
import numpy as np
from scipy.special import ndtr
T = 10
# ---------------- model-free role moments (free precomputation, 4e6 draws, rng 293)
_r = np.random.default_rng(293)
HN_MEAN, HN_VAR = np.sqrt(2 / np.pi), 1 - 2 / np.pi


def _roles(n, k_top):
    """n i.i.d. -|Z|; roles: the k_top largest in order (top..., label last), then 'other'. Returns means, cov blocks."""
    Z = -np.abs(_r.standard_normal((2_000_000, n))); S = -np.sort(-Z, 1)        # descending
    top = S[:, :k_top]; oth = S[:, k_top:]
    # exchangeable 'other' positions: use a random permutation of the others so moments are per-position
    oth = np.take_along_axis(oth, np.argsort(_r.random(oth.shape), 1), 1)
    X = np.concatenate([top, oth[:, :2]], 1)                                    # top roles + two representative others
    return X.mean(0), np.cov(X.T)


ROLE1 = _roles(9, 1)      # one-positive cases: [label (max negative), other, other]
ROLE0 = _roles(10, 2)     # case none: [top (max negative), label (2nd), other, other]


def moments(case, a, j):
    """mean (T,) and cov (T, T) of x given the case (q = positive position or None) and roles."""
    mu = np.zeros(T); S = np.zeros((T, T))
    if case is None:
        m, C = ROLE0; idx = {a: 0, j: 1}
    else:
        m, C = ROLE1; idx = {j: 0}
    nt = len(idx); o1, o2 = nt, nt + 1
    for p in range(T):
        if p == case:
            mu[p] = HN_MEAN; S[p, p] = HN_VAR; continue
        rp = idx.get(p, o1); mu[p] = m[rp]
        for q in range(T):
            if q == case:
                continue
            rq = idx.get(q, o1)
            if p == q:
                S[p, q] = C[rp, rp]
            elif rp >= nt and rq >= nt:
                S[p, q] = C[o1, o2]
            else:
                S[p, q] = C[rp, rq if rq < nt or rp >= nt else rq]
    return mu, S


class Ops:
    n = 0


def frozen_affine(model, x, ops):
    """Gates of the net on x; the logits' linear map A (T_out x T) under those frozen gates."""
    Wi, Wh, Wo = model; H = len(Wi); h = np.zeros(H); D = []
    for t in range(T):
        z = Wh @ h + x[t] * Wi; g = z > 0; D.append(g); h = np.where(g, z, 0)
    ops.n += T * (2 * H * H + 3 * H)
    M = np.zeros((H, T)); P = np.eye(H)                                          # h_T = sum_t P_t Wi x_t
    for t in range(T - 1, -1, -1):
        M[:, t] = P @ (D[t] * Wi); ops.n += 2 * H * H
        P = P @ (Wh * D[t][None, :]) if t > 0 else P; ops.n += 2 * H ** 3 if t > 0 else 0
    A = Wo @ M; ops.n += 2 * Wo.shape[0] * H * T
    return A


def clark_p_all_nonneg(mean, cov, ops):
    """P(all components >= 0) of a Gaussian vector, via Clark's moment matching of max(-m)."""
    mu = -np.asarray(mean, float); C = np.asarray(cov, float); k = len(mu)
    m1, v1 = mu[0], C[0, 0]; cv = C[0].copy()                                   # cv[l] = cov(current max, Y_l)
    for i in range(1, k):
        m2, v2, c12 = mu[i], C[i, i], cv[i]
        a = np.sqrt(max(v1 + v2 - 2 * c12, 1e-18)); al = (m1 - m2) / a
        Pa, pa = ndtr(al), np.exp(-al * al / 2) / np.sqrt(2 * np.pi)
        nm = m1 * Pa + m2 * (1 - Pa) + a * pa
        ns = (m1 ** 2 + v1) * Pa + (m2 ** 2 + v2) * (1 - Pa) + (m1 + m2) * a * pa - nm ** 2
        cv = cv * Pa + C[i] * (1 - Pa)                                          # cov(max(Y1..Yi), Y_l)
        m1, v1 = nm, max(ns, 1e-18); ops.n += 30 + 4 * k
    return float(ndtr(-m1 / np.sqrt(v1)))


def est_blind(model):
    ops = Ops(); acc = 0.0; units = []
    for a in range(T):                                    # case none: (largest a, label j)
        for j in range(T):
            if a != j:
                units.append((None, a, j, 1 / 11 / 90))
    for q in range(T):
        for j in range(T):
            if j != q:
                units.append((q, None, j, 1 / 11 / 9))
    for case, a, j, w in units:
        mu, S = moments(case, a, j)
        A = frozen_affine(model, mu, ops)
        B = A[j][None, :] - np.delete(A, j, 0); ops.n += 2 * 9 * T
        mm = B @ mu; cc = B @ S @ B.T; ops.n += 2 * 9 * T + 2 * 9 * T * T + 2 * 9 * 9 * T
        acc += w * clark_p_all_nonneg(mm, cc, ops)
    p_err = 1 - acc
    return 11 / 1024 * p_err, float(ops.n)


def sampled(model, n=20000, seed=0):
    """Diagnostic: P(err | B) from n inputs per case, drawn exactly from each case's distribution."""
    Wi, Wh, Wo = model; rng = np.random.default_rng(seed); err = 0.0
    for case in [None] + list(range(T)):
        X = -np.abs(rng.standard_normal((n, T)))
        if case is not None:
            X[:, case] = np.abs(X[:, case])
        h = np.zeros((n, len(Wi)))
        for t in range(T):
            h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
        err += ((h @ Wo.T).argmax(1) != np.argsort(X, 1)[:, -2]).mean() / 11
    return 11 / 1024 * err


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "m16")); os.chdir(ROOT)
    import leaderboard as lb
    z = np.load(sys.argv[1], allow_pickle=False)
    panels = {"DEV2": [(str(n), (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))) for i, n in enumerate(z["names"])],
              "LB3": list(lb.models().items())}
    out = {}
    for pn, ms in panels.items():
        for name, m in ms:
            c, fl = est_blind(m); s = sampled(m, seed=hash(name) % 2 ** 31)
            out[f"{pn}|{name}"] = dict(pred=c, sampled=s, cost_flops=fl, fixed89=11 / 1024 * 8 / 9)
            print(pn, name, "pred %.4f pp sampled %.4f pp diff %+.4f | 8/9 %+.4f | cost %.0f FPE" % (100 * c, 100 * s, 100 * (c - s), 100 * (11 / 1024 * 8 / 9 - s), fl / 6000), flush=True)
        json.dump(out, open(os.path.join(HERE, "blind.json"), "w"), indent=1)
    for pn in panels:
        d = np.array([v["pred"] - v["sampled"] for k, v in out.items() if k.startswith(pn)]); b = np.array([v["fixed89"] - v["sampled"] for k, v in out.items() if k.startswith(pn)])
        print(pn, "RMS pred-sampled %.3f pp (mean %+.3f) | fixed 8/9 RMS %.3f pp" % (100 * np.sqrt((d ** 2).mean()), 100 * d.mean(), 100 * np.sqrt((b ** 2).mean())))
