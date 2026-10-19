"""T449 step 1 (measurement) and the candidate-bound checks.

usage: python measure.py PART MODEL [MODEL ...]   (PART: err | mod | vitali | oracle); rows -> out/PART.jsonl
  err    : frozen-rule estimates at 64k / 1e6 / 4e6 FPE and a reference at 1.6e7 FPE with an independent rule
           (RULE_SEED 27182); per-stratum terms M_h * mean g_h.
  mod    : modulus of continuity: |g(u + delta v) - g(u)| for 2,048 frozen nodes per stratum, random unit v in the
           9 within-configuration coordinates (u_0 only picks the configuration), delta = 1e-1 .. 1e-6; also the
           angle between the two lines' face directions w = y/|y|, and second differences (L2 smoothness exponent).
  vitali : 3-D Vitali variation sum |Delta^3 g| of g on grids n = 4..64 in coordinates (1, 2, 3) with the others
           fixed at a node, on the full cube and on the interior box [.25, .75]^3 (Hardy-Krause needs it bounded as n grows).
  oracle : lower bound on the width of ANY cell-enclosure bound: dyadic cells (configuration x 2^k boxes),
           width >= sum_cells m (max - min of g over 32 points in the cell); cells subsampled, 95% CI.
Perturbation directions and in-cell points use numpy rng seeds fixed below (measurement only, not part of a bound).
"""
import json
import os
import sys
import time

import numpy as np

import lib

OUT = os.path.join(lib.HERE, "out")
os.makedirs(OUT, exist_ok=True)
DELTAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
N0 = 2048


def write(part, row):
    with open(os.path.join(OUT, part + ".jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")


def part_err(name, m):
    row = dict(model=name)
    for B in (64_000, 1_000_000, 4_000_000):
        e, c, det = lib.E.est(m, B, None, details=True)
        row[f"est_{B}"] = e; row[f"cost_{B}"] = c
        row[f"strata_{B}"] = [det[s][0] for s in lib.STRATA]
    e, c, det = lib.E.est(m, 16_000_000, None, rule_seed=27_182, details=True)
    row["ref"] = e; row["ref_cost"] = c; row["ref_strata"] = [det[s][0] for s in lib.STRATA]
    write("err", row)


def clip(U):
    return np.clip(U, 1e-12, 1 - 1e-12)


def wdir(j, U):
    kind, h = lib.STRATA[j]
    y = lib.E.face_points_strat(lib.T, kind, h, U)[0]
    return y / np.linalg.norm(y, axis=1, keepdims=True)


def part_mod(name, m):
    rng = np.random.default_rng(449)
    for j in range(len(lib.STRATA)):
        U = lib.node(j, N0)
        g0, c0 = lib.g(m, j, U)
        w0 = wdir(j, U)
        V = rng.standard_normal((N0, lib.T)); V[:, 0] = 0
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        row = dict(model=name, stratum=j, M=lib.MASS[j], mean_g=float(g0.mean()), frac_pos=float((g0 > 0).mean()),
                   cost_per_line=float(c0.mean()), d={})
        for dl in DELTAS:
            U1 = clip(U + dl * V); U2 = clip(U - dl * V)
            g1, _ = lib.g(m, j, U1); g2, _ = lib.g(m, j, U2)
            d1 = np.abs(g1 - g0); s2 = np.abs(g1 - 2 * g0 + g2)
            ang = np.arccos(np.clip((w0 * wdir(j, U1)).sum(1), -1, 1))
            sl = d1 / dl
            q = lambda x: [float(np.quantile(x, p)) for p in (.5, .9, .99)] + [float(x.max()), float(x.mean())]
            row["d"][str(dl)] = dict(abs=q(d1), slope=q(sl), sec_l2=float(np.sqrt((s2 ** 2).mean())),
                                     fd_l2=float(np.sqrt((d1 ** 2).mean())), ang_med=float(np.median(ang)),
                                     slope_ang=q(d1 / np.maximum(ang, 1e-300)),
                                     top=[int(i) for i in np.argsort(-sl)[:5]],
                                     frac_jump=float((d1 > 1e-3).mean()))
            if dl == 1e-6:
                np.save(os.path.join(OUT, f"slope_{name}_{j}.npy"), np.c_[sl, d1 / np.maximum(ang, 1e-300), g0])
        write("mod", row)


def part_vitali(name, m, anchors=3):
    for j in (0, 2, 3):
        for a in range(anchors):
          for box in ((0.0, 1.0), (0.25, 0.75)):
            base = lib.node(j, 1, lo=17 + 101 * a)[0]
            row = dict(model=name, stratum=j, anchor=a, box=box, V3={})
            for n in (4, 8, 16, 32, 64):
                gr = box[0] + (box[1] - box[0]) * (np.arange(n + 1) + .5) / (n + 1)
                G = np.stack(np.meshgrid(gr, gr, gr, indexing="ij"), -1).reshape(-1, 3)
                U = np.repeat(base[None], len(G), 0); U[:, 1:4] = G
                v = lib.g(m, j, U)[0].reshape(n + 1, n + 1, n + 1)
                d3 = np.diff(np.diff(np.diff(v, axis=0), axis=1), axis=2)
                d2 = np.diff(np.diff(v[:, :, 0], axis=0), axis=1)
                row["V3"][str(n)] = [float(np.abs(d3).sum()), float(np.abs(d2).sum())]
            write("vitali", row)


def part_oracle(name, m, ncell=1500, npt=32):
    rng = np.random.default_rng(4490)
    for j in range(len(lib.STRATA)):
        kind, _ = lib.STRATA[j]
        ncfg = len(lib.C.configs(lib.T, kind))
        row = dict(model=name, stratum=j, M=lib.MASS[j], ncfg=ncfg, k={})
        for k in (0, 3, 6, 9, 12, 18, 27, 36):
            per = np.array([k // 9 + (1 if i < k % 9 else 0) for i in range(9)])   # splits per coordinate 1..9
            side = 2.0 ** -per
            cfg = rng.integers(0, ncfg, ncell)
            lo = np.floor(rng.random((ncell, 9)) / side) * side
            P = rng.random((ncell, npt, 9)) * side + lo[:, None]
            U = np.empty((ncell * npt, lib.T))
            U[:, 0] = np.repeat((cfg + .5) / ncfg, npt); U[:, 1:] = clip(P.reshape(-1, 9))
            v = lib.g(m, j, U)[0].reshape(ncell, npt)
            r = v.max(1) - v.min(1)
            boot = [r[rng.integers(0, ncell, ncell)].mean() for _ in range(400)]
            row["k"][str(k)] = dict(ncells=float(ncfg * 2.0 ** k), width=float(lib.MASS[j] * r.mean()),
                                    lo=float(lib.MASS[j] * np.quantile(boot, .025)),
                                    hi=float(lib.MASS[j] * np.quantile(boot, .975)),
                                    frac_zero=float((r == 0).mean()))
        write("oracle", row)


if __name__ == "__main__":
    part, names = sys.argv[1], sys.argv[2:]
    ms, _ = lib.models()
    fn = dict(err=part_err, mod=part_mod, vitali=part_vitali, oracle=part_oracle)[part]
    for n in names:
        t = time.time()
        fn(n, ms[n])
        print(part, n, f"{time.time() - t:.0f}s", flush=True)
