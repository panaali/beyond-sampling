"""DIAGNOSTIC ORACLE (sampling; DRI request after T294): how much of rank_exact's per-class bias would remain if each
class's gates were not frozen at one pattern but closed by a mean-field / moment-matching treatment?
Class c = (top-3 triple j, K = #positive inputs), the 7,920 rank_exact classes; inputs from rank_exact's own law (ranks
4-10 in the triple's fixed arrangement, see draw), so 'net' differs from the truth only by rank_exact's design bias.
Per model:
  net     P(correct | class) of the network, sampled (half B);
  frozen  the same inputs through gates frozen at rank_exact's canonical class input (rank_exact's Q -> inf limit);
  gauss   ORACLE Gaussian margin law: the 9 rival logit margins of the class get their exact (sampled, half A) mean and
          covariance, and P(all margins > 0) is taken under that Gaussian. This is the best any Gaussian moment-matching
          closure (switching probabilities instead of frozen gates) can do at class level: its residual is the
          non-Gaussianity of the margin law, not moment error.
Class masses P(K)/720. Bias = surrogate - net. Per-class RMS includes sampling noise (cls_rms_noise).
usage: python meanfield_oracle.py [n] -> results/meanfield_oracle.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf  # noqa: E402
sys.path.insert(0, os.path.join(cf.HERE, "..", "rank_exact"))
import rank_est as R  # noqa: E402

TIDX = np.full((10, 10, 10), -1)
for j, (a, b, c) in enumerate(R.ORDS[:, :3]):
    TIDX[a, b, c] = j


def draw(rng, n):
    """Inputs from rank_exact's own law: i.i.d. N(0,1) values, sorted, placed by the fixed arrangement ORDS[j] of a
    uniform top-3 triple j (rank_exact integrates over exactly this law). -> x, class = j * 11 + K."""
    z = -np.sort(-rng.standard_normal((n, 10)), 1); j = rng.integers(0, 720, n)
    x = np.empty_like(z); x[np.arange(n)[:, None], R.ORDS[j]] = z
    return x, j * 11 + (z > 0).sum(1)


def net_gates(model, x):
    Wi, Wh, Wo = model; h = np.zeros((len(x), 16)); G = np.empty((len(x), 10, 16), bool)
    for t in range(10):
        z = h @ Wh.T + np.outer(x[:, t], Wi); G[:, t] = z > 0; h = np.where(G[:, t], z, 0)
    return (h @ Wo.T).argmax(1), G


def masked(model, x, M):
    Wi, Wh, Wo = model; h = np.zeros((len(x), 16))
    for t in range(10):
        h = M[:, t] * (h @ Wh.T + np.outer(x[:, t], Wi))
    return (h @ Wo.T).argmax(1)


def canon_gates(model):
    rank_of_pos = np.argsort(R.ORDS, 1); out = np.empty((720, 11, 10, 16))
    for k in range(11):
        _, G = net_gates(model, R.EOSK[k][rank_of_pos]); out[:, k] = G
    return out.reshape(7920, 10, 16)


def logits(model, x):
    Wi, Wh, Wo = model; h = np.zeros((len(x), 16))
    for t in range(10):
        h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0)
    return h @ Wo.T


def one(model, name, n=400_000, chunk=50_000, ngauss=4000):
    rng = np.random.default_rng(294_000 + sum(map(ord, name)))
    tgt = R.ORDS[:, 1]; oth = np.array([[p for p in range(10) if p != t] for t in tgt])     # (720, 9)
    S1 = np.zeros((7920, 9)); S2 = np.zeros((7920, 9, 9)); N = np.zeros(7920)
    for _ in range(n // 2 // chunk):                     # half A: per-class moments of the 9 rival margins
        x, c = draw(rng, chunk); L = logits(model, x); j = c // 11
        D = L[np.arange(chunk), tgt[j]][:, None] - L[np.arange(chunk)[:, None], oth[j]]
        np.add.at(S1, c, D); np.add.at(S2, c, D[:, :, None] * D[:, None, :]); np.add.at(N, c, 1)
    # sparse classes shrink to the triple's K-pooled moments (prior weight 30 samples)
    jj = np.arange(7920) // 11
    T1 = np.zeros((720, 9)); T2 = np.zeros((720, 9, 9)); TN = np.zeros(720)
    np.add.at(T1, jj, S1); np.add.at(T2, jj, S2); np.add.at(TN, jj, N)
    w0 = 30.0
    M1 = (S1 + w0 * (T1 / TN[:, None])[jj]) / (N + w0)[:, None]
    M2 = (S2 + w0 * (T2 / TN[:, None, None])[jj]) / (N + w0)[:, None, None]
    Cov = M2 - M1[:, :, None] * M1[:, None, :] + 1e-9 * np.eye(9)
    Lc = np.linalg.cholesky(Cov)
    zg = np.random.default_rng(7).standard_normal((ngauss, 9))           # model-free Gaussian nodes
    pg = np.array([((M1[c] + zg @ Lc[c].T) > 0).all(1).mean() for c in range(7920)])
    FG = canon_gates(model)
    ok_n = np.zeros(7920); ok_f = np.zeros(7920); cls_n = np.zeros(7920)
    for _ in range(n // 2 // chunk):                     # half B: paired network vs frozen-gate surrogate
        x, c = draw(rng, chunk); y = np.argsort(-x, 1)[:, 1]
        yn, _ = net_gates(model, x)
        np.add.at(ok_n, c, (yn == y)); np.add.at(ok_f, c, masked(model, x, FG[c]) == y); np.add.at(cls_n, c, 1)
    PK = R.PK[np.arange(7920) % 11] / 720               # class masses under rank_exact's law
    nz = cls_n > 0
    an = np.where(nz, ok_n / np.maximum(cls_n, 1), 0); af = np.where(nz, ok_f / np.maximum(cls_n, 1), 0)
    out = dict(net=float((PK * an).sum() / PK[nz].sum()), frozen=float((PK * af).sum() / PK[nz].sum()),
               gauss=float((PK * pg)[nz].sum() / PK[nz].sum()))
    # mass-weighted RMS of per-class bias (half-B class means; includes sampling noise)
    out["cls_rms_frozen"] = float(np.sqrt((PK[nz] * (af - an)[nz] ** 2).sum() / PK[nz].sum()))
    out["cls_rms_gauss"] = float(np.sqrt((PK[nz] * (pg - an)[nz] ** 2).sum() / PK[nz].sum()))
    out["cls_rms_noise"] = float(np.sqrt((PK[nz] * (an * (1 - an) / np.maximum(cls_n, 1))[nz]).sum() / PK[nz].sum()))
    blind = (np.arange(7920) % 11) <= 1
    for lab, sel in (("blind", blind), ("bulk", ~blind)):
        s_ = nz & sel
        for m, a in (("frozen", af), ("gauss", pg)):
            out[f"{lab}_bias_{m}"] = float((PK * (a - an))[s_].sum())     # contribution to the accuracy bias
    return out


def main(n=400_000):
    path = os.path.join(cf.HERE, "results", "meanfield_oracle.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    ms = {**cf.lb3_models(), **cf.dev2_models()}
    for name, m in ms.items():
        if name in res:
            continue
        res[name] = one(m, name, n)
        r = res[name]
        print(name, f"net {r['net']:.4f}", " ".join(f"{k} {100 * (r[k] - r['net']):+.2f}" for k in ("frozen", "gauss")),
              f"cls_rms frozen {r['cls_rms_frozen']:.3f} gauss {r['cls_rms_gauss']:.3f} noise {r['cls_rms_noise']:.3f}", flush=True)
        json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400_000)
