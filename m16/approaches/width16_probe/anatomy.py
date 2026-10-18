"""T206: cost anatomy of estT exact mode (unchanged estT.py copy; instrumentation by wrapping mass/_cell_error only).
usage: python anatomy.py H T name [start]   start = chamber (default, 4*T! half-chambers as estT) | truth (2T(T-1)
truth cones {x_i > x_j > rest} x sign(x0), the T192 / review/chamber_split lossless start; exact mode only)
-> anatomy/H{H}_T{T}_<name>_<start>.json (summary) + .npz (per-leaf: err, mass_ops, simplices, kind, leaf_mass)
Per leaf: kind 0 = fully correct (no clip leaves a wrong piece, so no mass computed), 1 = mixed, 2 = fully wrong.
leaf_mass (the leaf's own Gaussian mass) is diagnostic, computed with a throwaway counter (not in ops)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo
from leaderboard import _noisy
H, T, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; START = sys.argv[4] if len(sys.argv) > 4 else "chamber"
fn = os.path.join(HERE, "anatomy", f"H{H}_T{T}_{name}_{START}")
m = zoo(H, T, int(name[3])) if "noisy" not in name else _noisy(zoo(H, T, 0), 4, 0.005, 7)[int(name[-1])]


def truth_cones(T):
    E = np.eye(T); out = []
    for sigma in (1.0, -1.0):
        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                N = [E[i] - E[j]] + [E[j] - E[k] for k in range(T) if k not in (i, j)] + [sigma * E[0]]
                out.append(dict(sigma=sigma, perm=(i, j), truth=j, n_true=E[i] - E[j], cone=estT.Cone.from_normals(np.array(N))))
    return out


if START == "truth":
    estT._HALVES[T] = truth_cones(T)
MASS = [0]; mass0 = estT.mass


def mass(c, ops):
    o2 = Ops(); v = mass0(c, o2); ops.add(o2.n); MASS[0] += o2.n; return v


estT.mass = mass
LEAF = []; ce0 = estT._cell_error


def ce(c, g, A, Wo, truth, ops):
    m0, s0 = MASS[0], estT.STATS["simplices"]
    e = ce0(c, g, A, Wo, truth, ops)
    mo, ns = MASS[0] - m0, estT.STATS["simplices"] - s0
    s1 = estT.STATS["simplices"]
    lm = mass0(c, Ops()); estT.STATS["simplices"] = s1          # diagnostic leaf mass, not counted
    kind = 0 if mo == 0 else (2 if abs(e - lm) <= 1e-12 * max(lm, 1e-300) + 1e-18 else 1)
    LEAF.append((e, mo, ns, kind, lm, len(c.R))); return e


estT._cell_error = ce
estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0)
o = Ops(); t0 = time.process_time(); acc = estT.estT(m, o, mode="exact")
L = np.array(LEAF); err, mo, ns, kind, lm, nr = L.T
tot_err = err.sum(); thr = {}
for th in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
    s = (err > 0) & (err < th)
    thr[f"{th:.0e}"] = dict(cells=int(s.sum()), err_mass=float(err[s].sum()), err_frac=float(err[s].sum() / tot_err),
                           mass_ops=float(mo[s].sum()), mass_ops_frac=float(mo[s].sum() / MASS[0]))
r = dict(acc=acc, ops=o.n, mass_ops=MASS[0], geometry_ops=o.n - MASS[0], mass_frac=MASS[0] / o.n, cpu=time.process_time() - t0,
         leaves=len(L), leafsum_minus_1=float(lm.sum() - 1), simplices=int(ns.sum()),
         simplices_per_leaf=float(ns.mean()), simplices_per_massleaf=float(ns[mo > 0].mean()),
         mass_ops_per_simplex=MASS[0] / max(ns.sum(), 1), rays_per_leaf=float(nr.mean()),
         kinds={k: dict(cells=int((kind == i).sum()), err_mass=float(err[kind == i].sum()), mass_ops=float(mo[kind == i].sum()),
                        leaf_mass=float(lm[kind == i].sum())) for i, k in enumerate(("correct", "mixed", "wrong"))},
         error_below=thr, err_total=float(tot_err), **{k: v for k, v in estT.STATS.items() if k != "simplices"})
os.makedirs(os.path.dirname(fn), exist_ok=True); json.dump(r, open(fn + ".json", "w"), indent=1)
np.savez_compressed(fn + ".npz", err=err, mass_ops=mo, simplices=ns, kind=kind, leaf_mass=lm, rays=nr)
print(H, T, name, START, json.dumps(r)[:600], flush=True)
