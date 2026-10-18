"""T295 (pre-registered in README): per-unit readout diagnostic. (i) exact Gaussian orthant (Genz) and Clark on the exact sampled
margin moments; (ii) T294b mean-field state through step 10-m + exact last m steps (oracle draws from the state), with gate-pattern
branch counts. usage (repo root): python m16/approaches/meanfield/readout/readout.py DEV2_WEIGHTS.npz -> readout.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "blind11"), os.path.join(ROOT, "m16")]
import numpy as np
from scipy.stats import multivariate_normal as MVN
import blind as B11, mf2                  # T293 role moments / Clark, T294b closure pieces (read-only reuse)
T, H = 10, 16
MODELS = [0, 5, 20, 40, 60]; DIAG = [(None, 0, 1), (None, 7, 3), (3, None, 5), (9, None, 0), (0, None, 8)]
NREF, NOR = 400_000, 200_000


def sample(rng, case, a, j, n):
    if case is None:
        Zs = -np.sort(np.abs(rng.standard_normal((n, 10))), 1); X = np.empty((n, 10)); X[:, a] = Zs[:, 0]; X[:, j] = Zs[:, 1]
        rest = [p for p in range(10) if p not in (a, j)]; src = Zs[:, 2:]
    else:
        Zs = -np.sort(np.abs(rng.standard_normal((n, 9))), 1); X = np.empty((n, 10)); X[:, case] = np.abs(rng.standard_normal(n))
        X[:, j] = Zs[:, 0]; rest = [p for p in range(10) if p not in (case, j)]; src = Zs[:, 1:]
    X[:, rest] = np.take_along_axis(src, np.argsort(rng.random((n, 8)), 1), 1)
    return X


def run(m, h, X, t0):
    """exact steps t0..T-1 from h; returns logits and the gate bits (n, 16 (T - t0))."""
    Wi, Wh, Wo = m; G = []
    for t in range(t0, T):
        z = h @ Wh.T + X[:, t:t + 1] * Wi[None]; G.append(z > 0); h = np.maximum(z, 0)
    return h @ Wo.T, (np.concatenate(G, 1) if G else np.zeros((len(h), 0), bool))


def state(model, case, a, j, t_stop, variant):
    """T294b closure (variant 'a' or 'c') through steps 0..t_stop-1 -> mean, cov of v = (x, h_{t_stop})."""
    Wi, Wh, Wo = model; n = T + H; exact = variant in ("a", "c"); binned = variant in ("b", "c"); ops = mf2.Ops()
    mu_x, S_x = B11.moments(case, a, j); laws = mf2.laws_for(case, a, j) if binned else None
    mu = np.r_[mu_x, np.zeros(H)]; S = np.zeros((n, n)); S[:T, :T] = S_x
    for t in range(t_stop):
        K_ = np.zeros((H, n)); K_[:, T:] = Wh; K_[:, t] += Wi
        if laws is None:
            mz = K_ @ mu; Czv = K_ @ S; Cz = Czv @ K_.T; mh, Chh, Chv = mf2.relu_step(mz, Cz, Czv, exact, ops, ""); Shv = Chv[:, :T]
        else:
            bm, bv = laws[t]; g = S[:, t]; stt = S[t, t]; K = len(bm)
            muk = mu[None] + np.outer(bm - mu[t], g / stt); Sk = S[None] - np.outer(g, g)[None] * ((1 - bv / stt) / stt)[:, None, None]
            mz = muk @ K_.T; Czv = np.einsum("hn,knm->khm", K_, Sk); Cz = np.einsum("khm,gm->khg", Czv, K_)
            mhk, Chhk, Chvk = mf2.relu_step(mz, Cz, Czv, exact, ops, "")
            mh = mhk.mean(0); dh = mhk - mh; dx = muk[:, :T] - mu[None, :T]
            Chh = Chhk.mean(0) + dh.T @ dh / K; Shv = Chvk[:, :, :T].mean(0) + dh.T @ dx / K
        S2 = np.zeros((n, n)); S2[:T, :T] = S_x; S2[T:, :T] = Shv; S2[:T, T:] = Shv.T; S2[T:, T:] = Chh
        mu = np.r_[mu[:T], mh]; S = S2
    return mu, S


def psd_draw(rng, mu, S, n):
    w, V = np.linalg.eigh((S + S.T) / 2); A = V * np.sqrt(np.clip(w, 0, None)); return mu + rng.standard_normal((n, len(mu))) @ A.T


def branch_stats(ok, G):
    if G.shape[1] == 0:
        return dict(n_patterns=1, N99=1, N999=1, p_top99=float(ok.mean()))
    key = np.packbits(G, 1); key = key.view(np.dtype((np.void, key.shape[1]))).ravel()
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True); order = np.argsort(-cnt); c = np.cumsum(cnt[order]) / len(key)
    N99 = int(np.searchsorted(c, 0.99) + 1); N999 = int(np.searchsorted(c, 0.999) + 1)
    keep = np.isin(inv, order[:N99])
    return dict(n_patterns=int(len(u)), N99=N99, N999=N999, p_top99=float(ok[keep].mean()))


def crit(p_hat, p):
    return bool(abs(p_hat - p) <= (0.01 if p < 0.05 else 0.2 * p))


if __name__ == "__main__":
    z = np.load(sys.argv[1], allow_pickle=False); rng_u = np.random.default_rng(295)
    allu = mf2.units(); out = []
    for mi in MODELS:
        model = (z["Wi"][mi].astype(float), z["Wh"][mi].astype(float), z["Wo"][mi].astype(float))
        sel = [(c, a, j, w) for (c, a, j, w) in allu if (c, a, j) in DIAG] if mi in (0, 5) else []
        rest = [u for u in allu if u not in sel]; pick = rng_u.choice(len(rest), 10 - len(sel), replace=False)
        sel = sel + [rest[i] for i in pick]
        for case, a, j, w in sel:
            rng = np.random.default_rng(1000 * mi + 7); X = sample(rng, case, a, j, NREF)
            L, _ = run(model, np.zeros((NREF, H)), X, 0); Bm = np.eye(10)[j][None, :] - np.delete(np.eye(10), j, 0); M = L @ Bm.T
            P = float((M >= 0).all(1).mean()); mm, CC = M.mean(0), np.cov(M.T); o = B11.Ops(); o.n = 0
            r = dict(model=int(mi), name=str(z["names"][mi]), case=case, a=a, j=j, w=w, P=P,
                     i_genz=float(MVN(mean=-mm, cov=CC, allow_singular=True, abseps=1e-5, maxpts=200000).cdf(np.zeros(9))),
                     i_clark=float(B11.clark_p_all_nonneg(mm, CC, o)))
            for var in ("c", "a"):
                for mstep in (0, 1, 2, 3):
                    t0 = T - mstep; mu, S = state(model, case, a, j, t0, var); orng = np.random.default_rng(29500 + mstep)
                    V = psd_draw(orng, mu, S, NOR)
                    Lg, Gg = run(model, V[:, T:], V[:, :T], t0); okg = (Lg.argmax(1) == j)
                    # ii-T: true x, h | x by Gaussian regression on the state
                    Xt = sample(orng, case, a, j, NOR); Sxx = S[:T, :T]; Shx = S[T:, :T]; R = Shx @ np.linalg.pinv(Sxx)
                    Ch = S[T:, T:] - R @ Shx.T; ht = (mu[T:] + (Xt - mu[:T]) @ R.T) + psd_draw(orng, np.zeros(H), Ch, NOR)
                    Lt, Gt = run(model, ht, Xt, t0); okt = (Lt.argmax(1) == j)
                    r[f"ii_{var}_m{mstep}_G"] = dict(P=float(okg.mean()), **branch_stats(okg, Gg))
                    r[f"ii_{var}_m{mstep}_T"] = dict(P=float(okt.mean()), **branch_stats(okt, Gt))
            out.append(r)
            print(mi, (case, a, j), "P %.4f genz %.4f clark %.4f" % (P, r["i_genz"], r["i_clark"]),
                  " ".join("m%d G %.3f T %.3f N99 %d" % (k, r[f"ii_c_m{k}_G"]["P"], r[f"ii_c_m{k}_T"]["P"], r[f"ii_c_m{k}_T"]["N99"]) for k in range(4)), flush=True)
            json.dump(out, open(os.path.join(HERE, "readout.json"), "w"), indent=1)
    # summary
    meths = ["i_genz", "i_clark"] + [f"ii_{v}_m{k}_{s}" for v in ("c", "a") for k in range(4) for s in "GT"]
    summ = {}
    for mth in meths:
        ph = np.array([u[mth] if isinstance(u[mth], float) else u[mth]["P"] for u in out]); P = np.array([u["P"] for u in out])
        ok = [crit(a_, b_) for a_, b_ in zip(ph, P)]
        per = {}
        for mi in MODELS:
            us = [k for k, u in enumerate(out) if u["model"] == mi]; W = np.array([out[k]["w"] for k in us])
            per[str(mi)] = float(100 * 11 / 1024 * (W * (ph[us] - P[us])).sum() / W.sum())
        s = dict(pass_frac=float(np.mean(ok)), median_abs=float(np.median(np.abs(ph - P))), contrib_err_pp=per)
        if mth.startswith("ii"):
            s.update({k: float(np.mean([u[mth][k] for u in out])) for k in ("n_patterns", "N99", "N999")})
            s["N99_max"] = int(max(u[mth]["N99"] for u in out)); s["N999_max"] = int(max(u[mth]["N999"] for u in out))
            pt = np.array([u[mth]["p_top99"] for u in out]); s["top99_max_abs_shift"] = float(np.abs(pt - ph).max())
            s["pass_frac_top99"] = float(np.mean([crit(a_, b_) for a_, b_ in zip(pt, P)]))
        summ[mth] = s; print(mth, {k: v for k, v in s.items() if k != "contrib_err_pp"}, {k: round(v, 3) for k, v in per.items()})
    json.dump(summ, open(os.path.join(HERE, "readout_summary.json"), "w"), indent=1)
