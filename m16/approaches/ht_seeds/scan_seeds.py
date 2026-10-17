"""T2: replicate sci_tao's H x T scan (E25) over the zoo's training seeds 0-4.

For each (H, T, seed) model, zoo_2nd_argmax(H, T, seed=seed):
  * Sobol face rule (sci_tao est.py, copied unchanged) with R rule seeds 700..700+R-1, n faces per kind;
  * optionally the same faces drawn i.i.d. (face-MC; removes the low-discrepancy benefit), R_iid seeds;
  * tail share = share of E[F^2] carried by the top 1% of faces (mean over Sobol rules);
  * median cancellation factor of the R23 tie margin (as sci_tao scan_HT.py);
  * an INDEPENDENT ground truth by plain sampling (separate RNG stream), sized so that its s.e. is at most
    ~0.3x the estimator's RMS error (capped at --nmax samples).
Raw per-rule estimates/costs are stored, so gains and CIs are recomputed in analyse.py.
usage: OMP_NUM_THREADS=1 python scan_seeds.py --cells 16:4 16:5 ... --seeds 0 1 2 3 4 [--R 16 --Riid 8 --n 1024]
"""
import argparse, json, os, time, zlib
import numpy as np
import core as C
from est import est_faces

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scan_seeds.json")


def jac(M, x):
    """Gated input Jacobian of the logits (copied from sci_tao/cancel_T.py)."""
    Wi, Wh, Wo = M; n, T = x.shape; H = len(Wi)
    h = np.zeros((n, H)); J = np.zeros((n, H, T))
    for t in range(T):
        z = h @ Wh.T + x[:, t:t + 1] * Wi[None]
        Jz = np.einsum("ij,njk->nik", Wh, J); Jz[:, :, t] += Wi[None]
        g = z > 0; h = np.where(g, z, 0); J = np.where(g[:, :, None], Jz, 0)
    return np.einsum("oi,nik->nok", Wo, J)


def faces_iid(M, n, seed):
    T = M[2].shape[0]; _, Z23, Z12 = C.tables(T); rng = np.random.default_rng(seed)
    err = 0; cost = 0; Fs = []
    for kind, Z in (("23", Z23), ("12", Z12)):
        if Z == 0: continue
        y, a, b, tau = C.face_points(T, kind, rng.random((n, T)))
        F, c, _ = C.walk_mass(M, y, a, b, tau, kind); err += Z * F.mean(); cost += c.sum(); Fs.append(F)
    return 1 - err, cost, Fs


def tail_share(Fs):
    F = np.concatenate(Fs); s = np.sort(F)[::-1]
    return float((s[:max(1, len(s) // 100)] ** 2).sum() / max((s ** 2).sum(), 1e-300))


def cancel(M, T):
    U = C.sobol(T, 2048, 9); y, a, b, _ = C.face_points(T, "23", U)
    k = y[np.arange(len(y)), a] > 0; y, a, b = y[k], a[k], b[k]; r_ = np.arange(len(y))
    J = jac(M, y + 1e-9 * (np.arange(T)[None] == a[:, None]))
    terms = (J[r_, a] - J[r_, b]) * y
    return float(np.median(np.abs(terms).sum(1) / np.abs(terms.sum(1))))


def truth(M, n, key, chunk=200_000):
    """Plain-sampling accuracy (numpy first-index argmax, label = argsort rank 2) on an RNG stream that no
    estimator uses."""
    T = M[2].shape[0]
    rng = np.random.default_rng(31_337 + zlib.crc32(key.encode()) % 100_000)
    ok = 0; done = 0
    while done < n:
        m = min(chunk, n - done)
        x = rng.standard_normal((m, T))
        ok += int((C.forward(M, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum()); done += m
    return ok, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--Riid", type=int, default=8)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--nmin", type=int, default=2_000_000)
    ap.add_argument("--nmax", type=int, default=32_000_000)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in a.cells:
        H, T = map(int, spec.split(":"))
        for sd in a.seeds:
            key = f"{H}:{T}:{sd}"
            row = out.get(key, {})
            M = C.zoo(H, T, seed=sd); t0 = time.time()
            if "sobol" not in row:
                e, cs, tl = [], [], []
                for r in range(a.R):
                    acc, c, o = est_faces(M, a.n, a.n, rule_seed=700 + r)
                    e.append(float(acc)); cs.append(float(c)); tl.append(tail_share(list(o.values())))
                row["sobol"] = dict(est=e, cost=cs, tail=tl, n=a.n)
            if a.Riid and "iid" not in row:
                e, cs, tl = [], [], []
                for r in range(a.Riid):
                    acc, c, Fs = faces_iid(M, a.n, 800 + r)
                    e.append(float(acc)); cs.append(float(c)); tl.append(tail_share(Fs))
                row["iid"] = dict(est=e, cost=cs, tail=tl, n=a.n)
            if "cancel" not in row:
                row["cancel"] = cancel(M, T)
            t1 = time.time()
            e = np.array(row["sobol"]["est"]); p = e.mean()
            rmse_guess = max(e.std(ddof=1), 1e-7)
            need = int(min(a.nmax, max(a.nmin, p * (1 - p) / (0.3 * rmse_guess) ** 2)))
            need = int(np.ceil(need / 200_000) * 200_000)
            if row.get("truth", {}).get("n", 0) < need:
                ok, n = truth(M, need, key)
                row["truth"] = dict(ok=ok, n=n, acc=ok / n, se=float(np.sqrt(ok / n * (1 - ok / n) / n)))
            out[key] = row
            json.dump(out, open(a.out, "w"), indent=1)
            g = p * (1 - p) / np.mean(row["sobol"]["cost"]) / e.var(ddof=1)
            print(f"H={H:2d} T={T:2d} seed={sd} truth={row['truth']['acc']:.5f}+-{row['truth']['se']:.1e} "
                  f"(n={row['truth']['n']:.1e}) Sobol mean={p:.5f} var-gain={g:.1f} "
                  f"tail={np.mean(row['sobol']['tail']):.2f} cancel={row['cancel']:.0f} "
                  f"[est {t1 - t0:.0f}s, truth {time.time() - t1:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
