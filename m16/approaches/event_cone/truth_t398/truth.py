"""T398 truths for E1(t0) = {pick = p3, v2 - v3 > t0} on orig. SAMPLING-CLASS (random model inputs), unbiased.
T362's conditional MC along exact v3 lines (src/cmc.py, branch claude/t286-cheap-frontier @535d5bde), with the
E1(1.0) code path unchanged except that t0 is a parameter. Runs in chunks; each chunk appends one JSON line to
runs/t<t0>_<first chunk>.jsonl (one file per process; pooled over
all files of that t0) and the driver stops at rel. s.e. <= TARGET or after max_sec of wall time (one core).
usage: python truth.py t0 first_chunk_index lines_per_chunk max_sec [target_rel_se]"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))          # m16/ (leaderboard)
import cf  # noqa: E402
import rare  # noqa: E402

EPS = 1e-9
SEED0 = 398000


def run_e1(model, t0, nlines, seed, batch=200_000):
    """cmc.run(model, "E1(1.0)", ...) with t0 passed in; every other line is T362's."""
    rng = np.random.default_rng(seed); Zs = []; ndrawn = 0; flops = 0.0; cells = 0
    need = nlines
    while need > 0:
        x = rng.standard_normal((batch, 10)); o = np.argsort(-x, 1); r = np.arange(batch)
        v = x[r[:, None], o]; v1, v2, v4 = v[:, 0], v[:, 1], v[:, 3]
        ok = (v2 - v4) > t0
        idx = np.flatnonzero(ok)
        if len(idx) >= need:                      # stop exactly at the need-th passing input
            cut = idx[need - 1] + 1; idx = idx[:need]
        else:
            cut = batch
        ndrawn += cut; need -= len(idx)
        Zs.append(np.zeros(cut - len(idx)))      # screened inputs: Z = 0 exactly (model-free)
        if not len(idx):
            continue
        xs = x[idx]; p2, p3 = o[idx, 1], o[idx, 2]; s = v2[idx] - EPS; lo4 = v4[idx]
        x0 = xs.copy(); x0[np.arange(len(idx)), p3] = s
        d = np.zeros_like(x0); d[np.arange(len(idx)), p3] = -1.0
        tmax = s - lo4
        pk3, pkc, f, it, nc = rare.walk3(model, x0, d, p2, p3, tmax); flops += f; cells += nc
        n = len(idx); mass = np.zeros(n)
        for rows, a, b in pk3:
            a = np.maximum(a, t0 - EPS); m = b > a
            np.add.at(mass, rows[m], ndtr(s[rows[m]] - a[m]) - ndtr(s[rows[m]] - b[m])); flops += m.sum() * (2 * rare.G.F_PHI + 2)
        Z = mass / (ndtr(v2[idx]) - ndtr(lo4)); flops += n * 4
        Zs.append(Z)
    Z = np.concatenate(Zs)
    return Z, ndrawn, flops, cells


def orig():
    import leaderboard as L
    return tuple(np.asarray(w, float) for w in L.models(16, 10)["orig"])


def summary(recs):
    """Pooled estimate over chunks: every drawn input is one i.i.d. Z (screened ones are 0)."""
    N = sum(r["n_drawn"] for r in recs); s1 = sum(r["sumZ"] for r in recs); s2 = sum(r["sumZ2"] for r in recs)
    est = s1 / N; var = (s2 - N * est ** 2) / (N - 1)
    return dict(est=est, se=float(np.sqrt(max(var, 0.0) / N)), n_drawn=N, n_lines=sum(r["n_lines"] for r in recs),
                n_nonzero=sum(r["n_nonzero"] for r in recs), sec=sum(r["sec"] for r in recs),
                C_fpe=sum(r["flops"] for r in recs) / cf.FPE, n_chunks=len(recs))


def load(t0):
    import glob
    return [json.loads(l) for p in sorted(glob.glob(os.path.join(HERE, "runs", f"t{t0}_*.jsonl"))) for l in open(p)]


def main(t0, i0, lines, max_sec, target=0.10):
    m = orig(); os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out = os.path.join(HERE, "runs", f"t{t0}_{i0}.jsonl"); t_start = time.time(); i = i0
    while True:
        t = time.time(); Z, nd, fl, cells = run_e1(m, t0, lines, SEED0 + i)
        nz = Z[Z > 0]
        rec = dict(t0=t0, chunk=i, seed=SEED0 + i, n_lines=lines, n_drawn=int(nd), sumZ=float(Z.sum()),
                   sumZ2=float((Z ** 2).sum()), n_nonzero=int(len(nz)), flops=fl, cells=int(cells),
                   sec=time.time() - t, Z_nonzero=nz.tolist())
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        s = summary(load(t0)); rel = s["se"] / s["est"] if s["est"] > 0 else np.inf
        print(f"t0={t0} chunk={i} est={s['est']:.4g} se={s['se']:.3g} rel={rel:.3f} nz={s['n_nonzero']} "
              f"lines={s['n_lines']} sec={s['sec']:.0f}", flush=True)
        i += 1
        if rel <= target or time.time() - t_start + rec["sec"] > max_sec:
            break


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), int(a[1]), int(a[2]), float(a[3]), float(a[4]) if len(a) > 4 else 0.10)
