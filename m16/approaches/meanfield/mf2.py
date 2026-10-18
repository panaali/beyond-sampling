"""T294b (pre-registered in README): stage-1 mean-field closure with (a) exact bivariate ReLU cross moments, (b) the sign-fixed
law of the current input (K = 16 equal-probability bins, collapsed by total covariance), (c) both. Same 180 blind units, role
moments and Clark readout as mf.py; reference = T293 sampled contributions. usage (repo root):
python m16/approaches/meanfield/mf2.py DEV2_WEIGHTS.npz [variants=a,b,c] -> mf2.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [os.path.join(HERE, "..", "blind11"), os.path.join(ROOT, "m16")]
import numpy as np
from scipy.special import ndtr
import blind as B11                      # role moments, Clark closure (T293, read-only reuse)
T, K = 10, 16
PHI = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
GX, GW = np.polynomial.legendre.leggauss(20)


class Ops:
    def __init__(self): self.d = {}
    def add(self, k, v): self.d[k] = self.d.get(k, 0) + v
    @property
    def n(self): return sum(self.d.values())


def phi2(a, b, rho):
    """Bivariate normal CDF (unit variances), 20-node Gauss-Legendre in theta = asin(rho)."""
    th = np.arcsin(rho); t = 0.5 * th[..., None] * (GX + 1); st, ct2 = np.sin(t), np.cos(t) ** 2
    f = np.exp(-(a[..., None] ** 2 + b[..., None] ** 2 - 2 * a[..., None] * b[..., None] * st) / (2 * ct2))
    return ndtr(a) * ndtr(b) + (0.5 * th / (2 * np.pi)) * (f * GW).sum(-1)


def e_relu_pair(a, b, rho):
    """E[relu(u + a) relu(v + b)] for standard (u, v) with corr rho."""
    r2 = 1 - rho ** 2; r = np.sqrt(r2); p2 = np.exp(-(a * a - 2 * rho * a * b + b * b) / (2 * r2)) / (2 * np.pi * r)
    return (a * b + rho) * phi2(a, b, rho) + a * PHI(b) * ndtr((a - rho * b) / r) + b * PHI(a) * ndtr((b - rho * a) / r) + r2 * p2


# ---------------- model-free role laws, binned (K equal-probability bins: mean, variance), 4e6 draws, rng 2941
def _bins(samples):
    q = np.sort(samples); c = np.array_split(q, K)
    return np.array([x.mean() for x in c]), np.array([x.var() for x in c])


_r = np.random.default_rng(2941)
_Z = -np.sort(np.abs(_r.standard_normal((4_000_000, 10))), 1)                    # case none, descending: top, label, others
BIN0 = {"top": _bins(_Z[:, 0]), "label": _bins(_Z[:, 1]), "other": _bins(_Z[np.arange(len(_Z)), _r.integers(2, 10, len(_Z))])}
_Z = -np.sort(np.abs(_r.standard_normal((4_000_000, 9))), 1)                     # one positive: label (max negative), others
BIN1 = {"label": _bins(_Z[:, 0]), "other": _bins(_Z[np.arange(len(_Z)), _r.integers(1, 9, len(_Z))]),
        "pos": _bins(np.abs(_r.standard_normal(4_000_000)))}
del _Z


def relu_step(mz, Cz, Czv, exact, ops, tag):
    """Batched over leading dims: mz (..., H), Cz (..., H, H), Czv (..., H, n). -> mh, Chh, Chv."""
    H = mz.shape[-1]; nb = int(np.prod(mz.shape[:-1])) if mz.ndim > 1 else 1
    s = np.sqrt(np.maximum(np.einsum("...ii->...i", Cz), 1e-24)); a = mz / s; Pa = ndtr(a); pa = PHI(a)
    mh = mz * Pa + s * pa; Eh2 = (mz ** 2 + s ** 2) * Pa + mz * s * pa; vh = np.maximum(Eh2 - mh ** 2, 0)
    Chv = Pa[..., :, None] * Czv; ops.add(tag + "relu_moments", nb * (40 * H + H * Czv.shape[-1]))
    if exact:
        iu = np.triu_indices(H, 1); rho = np.clip(Cz[..., iu[0], iu[1]] / (s[..., iu[0]] * s[..., iu[1]]), -0.9999, 0.9999)
        E = s[..., iu[0]] * s[..., iu[1]] * e_relu_pair(a[..., iu[0]], a[..., iu[1]], rho) - mh[..., iu[0]] * mh[..., iu[1]]
        Chh = np.zeros_like(Cz); Chh[..., iu[0], iu[1]] = E; Chh[..., iu[1], iu[0]] = E
        ops.add(tag + "phi2_pairs", nb * len(iu[0]) * 350)
    else:
        Chh = (Pa[..., :, None] * Pa[..., None, :]) * Cz; ops.add(tag + "relu_moments", nb * 2 * H * H)
    i = np.arange(H); Chh[..., i, i] = vh
    return mh, Chh, Chv


def propagate(model, mu_x, S_x, laws, exact, ops):
    """laws: None (Gaussian x_t, stage 1) or per-position (bin means, bin vars) for the current input."""
    Wi, Wh, Wo = model; H = len(Wi); n = T + H
    mu = np.r_[mu_x, np.zeros(H)]; S = np.zeros((n, n)); S[:T, :T] = S_x
    for t in range(T):
        K_ = np.zeros((H, n)); K_[:, T:] = Wh; K_[:, t] += Wi
        if laws is None:
            mz = K_ @ mu; Czv = K_ @ S; Cz = Czv @ K_.T; ops.add("cov_prop", 2 * H * n + 2 * H * n * n + 2 * H * H * n)
            mh, Chh, Chv = relu_step(mz, Cz, Czv, exact, ops, "")
            Shv = Chv[:, :T]
        else:
            bm, bv = laws[t]; g = S[:, t]; stt = S[t, t]
            # condition the state on x_t ~ N(bm_k, bv_k) by Gaussian regression, per bin
            muk = mu[None] + np.outer(bm - mu[t], g / stt)                          # (K, n)
            Sk = S[None] - np.outer(g, g)[None] * ((1 - bv / stt) / stt)[:, None, None]  # (K, n, n)
            ops.add("bins_condition", K * (2 * n + 2 * n * n))
            mz = muk @ K_.T; Czv = np.einsum("hn,knm->khm", K_, Sk); Cz = np.einsum("khm,gm->khg", Czv, K_)
            ops.add("cov_prop", K * (2 * H * n + 2 * H * n * n + 2 * H * H * n))
            mhk, Chhk, Chvk = relu_step(mz, Cz, Czv, exact, ops, "")
            # collapse: total mean / covariance over the K equal-weight bins (x block kept at its original moments)
            mh = mhk.mean(0); dh = mhk - mh; dx = muk[:, :T] - mu[None, :T]
            Chh = Chhk.mean(0) + dh.T @ dh / K; Shv = Chvk[:, :, :T].mean(0) + dh.T @ dx / K
            ops.add("bins_collapse", K * (H * H + H * T) * 3)
        S2 = np.zeros((n, n)); S2[:T, :T] = S_x; S2[T:, :T] = Shv; S2[:T, T:] = Shv.T; S2[T:, T:] = Chh
        mu = np.r_[mu[:T], mh]; S = S2
    Ho = np.zeros((Wo.shape[0], n)); Ho[:, T:] = Wo
    mL = Ho @ mu; CL = Ho @ S @ Ho.T; ops.add("readout", 2 * Wo.shape[0] * n * (n + Wo.shape[0]))
    return mL, CL


def units():
    u = [(None, a, j, 1 / 11 / 90) for a in range(T) for j in range(T) if a != j]
    return u + [(q, None, j, 1 / 11 / 9) for q in range(T) for j in range(T) if j != q]


def laws_for(case, a, j):
    if case is None:
        return [BIN0["top"] if p == a else BIN0["label"] if p == j else BIN0["other"] for p in range(T)]
    return [BIN1["pos"] if p == case else BIN1["label"] if p == j else BIN1["other"] for p in range(T)]


def est_blind_mf2(model, variant):
    exact = variant in ("a", "c"); binned = variant in ("b", "c"); ops = Ops(); acc = 0.0
    for case, a, j, w in units():
        mu_x, S_x = B11.moments(case, a, j)
        mL, CL = propagate(model, mu_x, S_x, laws_for(case, a, j) if binned else None, exact, ops)
        Bm = np.eye(len(mL))[j][None, :] - np.delete(np.eye(len(mL)), j, 0)
        o = B11.Ops(); o.n = 0
        acc += w * B11.clark_p_all_nonneg(Bm @ mL, Bm @ CL @ Bm.T, o); ops.add("clark", o.n + 2 * 9 * 10 * 10)
    return 11 / 1024 * (1 - acc), ops.d


def stats(p, s):
    b = np.polyfit(s, p, 1)[0]; b2 = np.polyfit(p, s, 1)[0]
    return dict(rms_pp=100 * float(np.sqrt(((p - s) ** 2).mean())), mean_pp=100 * float((p - s).mean()), corr=float(np.corrcoef(p, s)[0, 1]),
                slope_pred_on_sampled=float(b), slope_sampled_on_pred=float(b2), sd_ratio=float(p.std() / s.std()))


if __name__ == "__main__":
    os.chdir(ROOT); import leaderboard as lb
    ref = json.load(open(os.path.join(HERE, "..", "blind11", "blind.json")))
    z = np.load(sys.argv[1], allow_pickle=False); VS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["a", "b", "c"]
    panels = {"DEV2": [(str(n), (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))) for i, n in enumerate(z["names"])],
              "LB3": list(lb.models().items())}
    fn = os.path.join(HERE, "mf2.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    for v in VS:
        for pn, ms in panels.items():
            for name, m in ms:
                key = f"{v}|{pn}|{name}"
                if key in out:
                    continue
                c, d = est_blind_mf2(m, v); s = ref[f"{pn}|{name}"]["sampled"]
                out[key] = dict(pred=c, sampled=s, cost_fpe={k: x / 6000 for k, x in d.items()})
                print(key, "pred %.4f pp sampled %.4f pp diff %+.4f cost %.0f FPE" % (100 * c, 100 * s, 100 * (c - s), sum(d.values()) / 6000), flush=True)
            json.dump(out, open(fn, "w"), indent=1)
    summ = {}
    for v in VS:
        for pn in panels:
            ks = [k for k in out if k.startswith(f"{v}|{pn}|")]
            p = np.array([out[k]["pred"] for k in ks]); s = np.array([out[k]["sampled"] for k in ks]); st = stats(p, s)
            st["cost_fpe"] = {c: float(np.mean([out[k]["cost_fpe"].get(c, 0) for k in ks])) for c in out[ks[0]]["cost_fpe"]}
            st["cost_total_fpe"] = sum(st["cost_fpe"].values())
            if pn == "DEV2":
                st["go"] = bool(st["corr"] >= 0.7 and st["rms_pp"] <= 0.15); st["stop"] = bool(st["corr"] < 0.3)
            summ[f"{v}|{pn}"] = st; print(v, pn, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in st.items() if k != "cost_fpe"})
    json.dump(summ, open(os.path.join(HERE, "mf2_summary.json"), "w"), indent=1)
