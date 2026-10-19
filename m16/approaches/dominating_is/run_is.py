"""T436 pre-registered run (README 'Pre-registration').
usage: OMP_NUM_THREADS=1 python run_is.py points            -> points.json (pilot 1e7 seed 1, 200 descents)
       OMP_NUM_THREADS=1 python run_is.py is T0             -> is_<T0>.json (n = 1e6 seed 2; one extension to 1e7 seed 3)"""
import json, os, sys, time
import numpy as np
from scipy.stats import multivariate_normal  # noqa: F401  (not used; densities computed directly)
from net import event, logits, top3
from dp import descend

HERE = os.path.dirname(os.path.abspath(__file__))
US_PER_FPE = 4.0


def points():
    t = time.time(); rng = np.random.default_rng(1); best = []   # (kappa, x)
    for _ in range(10):
        X = rng.standard_normal((1_000_000, 10)); top, v = top3(X); g23 = v[:, 1] - v[:, 2]
        K = logits(X).argmax(1) == top[:, 2]; kap = np.where(K, g23 / np.linalg.norm(X, axis=1), -1)
        idx = np.argsort(-kap)[:200]; best += [(kap[i], X[i]) for i in idx]
    best.sort(key=lambda z: -z[0]); seeds = best[:200]; t_pilot = time.time() - t
    t = time.time(); ends = []
    for k, x0 in seeds:
        x, r2, tri, _, steps = descend(x0)
        ends.append(dict(x=x.tolist(), r2=r2, tri=[int(i) for i in tri], kappa=float(k), steps=steps))
    t_desc = time.time() - t
    ends.sort(key=lambda e: e["r2"]); pts = []
    for e in ends:
        if all(np.linalg.norm(np.array(e["x"]) - np.array(p["x"])) >= 0.05 for p in pts):
            pts.append(e)
    out = dict(pilot_fpe=10_000_000, pilot_sec=t_pilot, descent_sec=t_desc, n_seeds=len(seeds), points=pts)
    json.dump(out, open(os.path.join(HERE, "points.json"), "w"), indent=1)
    print("pilot %.0fs, descents %.0fs, %d distinct end points; best r2:" % (t_pilot, t_desc, len(pts)),
          [(round(p["r2"], 4), tuple(p["tri"])) for p in pts[:12]], flush=True)


def mixture(t0, A0=0.05):
    P = json.load(open(os.path.join(HERE, "points.json")))["points"]
    r2min = P[0]["r2"]; sel = [p for p in P if p["r2"] <= r2min + 2 * np.log(1000) / t0 ** 2][:30]
    mu = np.array([t0 * np.array(p["x"]) for p in sel]); lw = -t0 ** 2 * np.array([p["r2"] for p in sel]) / 2
    w = np.exp(lw - lw.max()); w /= w.sum()
    return mu, w, sel


def log_q(X, mu, w, A0=0.05):
    # log of a0 N(0,I) + (1-a0) sum w_i N(mu_i, I), dropping the common (2 pi)^-5
    l0 = np.log(A0) - 0.5 * (X ** 2).sum(1)
    li = np.log((1 - A0) * w)[None] - 0.5 * ((X[:, None, :] - mu[None]) ** 2).sum(2)
    m = np.maximum(l0, li.max(1))
    return m + np.log(np.exp(l0 - m) + np.exp(li - m[:, None]).sum(1))


def is_run(t0, n, seed, mu, w, A0=0.05, chunk=200_000):
    rng = np.random.default_rng(seed); vals = []; fwd = 0; tri_hits = {}; nh = 0
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        comp = rng.choice(len(w) + 1, size=m, p=np.r_[A0, (1 - A0) * w])
        X = rng.standard_normal((m, 10)); X[comp > 0] += mu[comp[comp > 0] - 1]
        hit, top, k = event(X, t0); fwd += k
        lr = -0.5 * (X ** 2).sum(1) - log_q(X, mu, w, A0)
        v = np.where(hit, np.exp(lr), 0.0); vals.append(v); nh += int(hit.sum())
        for tr in map(tuple, top[hit]):
            tri_hits[str(tr)] = tri_hits.get(str(tr), 0) + 1
    v = np.concatenate(vals)
    return v, fwd, nh, tri_hits


def run(t0):
    P = json.load(open(os.path.join(HERE, "points.json")))
    mu, w, sel = mixture(t0); t = time.time()
    v, fwd, nh, th = is_run(t0, 1_000_000, 2, mu, w)
    ext = False
    p = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    if se / p > 0.1 if p > 0 else True:
        v2, f2, n2, th2 = is_run(t0, 10_000_000, 3, mu, w); ext = True
        v = np.concatenate([v, v2]); fwd += f2; nh += n2
        for k_, c in th2.items():
            th[k_] = th.get(k_, 0) + c
        p = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    sec = time.time() - t
    C = P["pilot_fpe"] + fwd + P["descent_sec"] * 1e6 / US_PER_FPE
    C_draws = P["pilot_fpe"] + len(v) + P["descent_sec"] * 1e6 / US_PER_FPE
    F = se ** 2 * C / (p * (1 - p)); F_draws = se ** 2 * C_draws / (p * (1 - p))
    out = dict(t0=t0, n=len(v), extended=ext, p=p, se=se, rel_se=se / p, hits=nh, forward=fwd, C=C, C_draws=C_draws,
               F=F, F_draws=F_draws, gain=1 / F, tri_hits=th, n_points=len(sel), weights=w.tolist(),
               point_r2=[s["r2"] for s in sel], point_tri=[s["tri"] for s in sel], sec=sec)
    json.dump(out, open(os.path.join(HERE, f"is_{t0}.json"), "w"), indent=1)
    print(f"t0 {t0}: p {p:.4e} se {se:.2e} (rel {se/p:.3f}) n {len(v)} hits {nh} fwd {fwd} C {C:.3e} F {F:.2e} "
          f"(gain {1/F:.2e}); F counting all draws {F_draws:.2e}; points {len(sel)}; hit triples {th}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "points":
        points()
    else:
        run(float(sys.argv[2]))
