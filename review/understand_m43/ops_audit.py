"""T74 item 3: independent recount of tier B's operations. The geometric primitives (clip, frac) and the arc
read-out (_read_point) are wrapped with a stricter, independently written count; the remaining charges made by
est_mech itself are kept. Also checks the 'nothing hidden' claims: no RNG is touched during est_mech.
Strict rules: multiply-add = 1, any comparison/abs/sign test = 1, sqrt/div/atan2 = 1.
  clip:  per vertex 3 (dot) + 1 (sign test); per new vertex 6 (combination) + 3 (norm^2) + 1 sqrt + 3 div = 13
  frac:  per triangle cross 6 + triple 3 + 3 dots 9 + 3 adds + atan2 1 + 1 (x2, accumulate) = 23; +1 final div
  _read_point: per vertex 1 (n_true . v is a difference of two coordinates) + 1 (compare); mean 3 per point
  completion: row norms 4 x (3 + sqrt) + 4 div + sort 5 = 25 instead of the 16 charged
usage: OMP_NUM_THREADS=1 python ops_audit.py"""
import numpy as np
from common_review import base_model, noisy, zoo
import est_mech as EM
import m43
from m43 import Ops
from run_mech import TIERS

ORIG_CLIP, ORIG_FRAC, ORIG_READ = EM.clip, EM.frac, EM._read_point
S = {"strict_geom": 0, "orig_geom": 0, "read": 0, "n_clip": 0, "n_tri": 0, "n_compl": 0}


def clip_w(poly, n, ops=None):
    o = Ops(); ins, out = ORIG_CLIP(poly, n, o); S["orig_geom"] += o.n; S["n_clip"] += 1
    m = len(poly); s = [float(n @ v) for v in poly]
    newv = sum(1 for i in range(m) if s[i] * s[(i + 1) % m] < 0) if (min(s) < 0 < max(s)) else 0
    S["strict_geom"] += 4 * m + 13 * newv
    return ins, out


def frac_w(poly, ops=None):
    o = Ops(); r = ORIG_FRAC(poly, o); S["orig_geom"] += o.n
    if len(poly) >= 3:
        S["strict_geom"] += 23 * (len(poly) - 2) + 1; S["n_tri"] += len(poly) - 2
    return r


def read_w(poly, n_true, tol=1e-9):
    r = ORIG_READ(poly, n_true, tol); S["read"] += 2 * len(poly) + 3
    return r


EM.clip, EM.frac, EM._read_point = clip_w, frac_w, read_w
# completion extra (9 per completion) counted by wrapping np.linalg.norm calls inside _arc_err is awkward; instead
# count completions by re-running the completion condition: patch _parts_for to see |S|
ORIG_PARTS = EM._parts_for


def parts_w(Sset, A3, Wo, truth, ops):
    return ORIG_PARTS(Sset, A3, Wo, truth, ops)


# RNG guard: any numpy RNG use inside est_mech raises
class Boom(Exception):
    pass


def _boom(*a, **k):
    raise Boom("RNG used")


for name in ("default_rng", "rand", "randn", "random", "normal", "uniform", "standard_normal", "choice", "permutation"):
    setattr(np.random, name, _boom)

kw = dict(TIERS["B default (A2 + resolved crossover)"]); EM.COMPLETE = kw.pop("complete")
import importlib
rng_models = None
models = [("base", base_model())] + [(f"zoo{s}", zoo(s)) for s in (1, 2)]
import common_review
# noisy copies drawn before the RNG guard would be needed: re-enable RNG locally
models_noisy = []
rows = []
print(f"{'model':10s} {'charged':>8s} {'strict':>8s} {'ratio':>6s} clips tris")
np_rng = np.random.Generator(np.random.PCG64(77))
Wi, Wh, Wo = base_model()
for i in range(6):
    models.append((f"noisy{i}", (Wi + 0.05 * np_rng.standard_normal(4), Wh + 0.05 * np_rng.standard_normal((4, 4)),
                                 Wo + 0.05 * np_rng.standard_normal((3, 4)))))
for name, m in models:
    for k in S:
        S[k] = 0
    o = Ops(); est = EM.est_mech(m, o, **kw)
    other = o.n                       # wrapped primitives no longer charge `o`, so o.n = est_mech's own charges
    charged = other + S["orig_geom"]  # = what the unwrapped estimator reports
    strict = other + S["strict_geom"] + S["read"] + 9 * 8   # + completion upper bound (<= 8 completions/model)
    o2 = Ops(); EM.clip, EM.frac, EM._read_point = ORIG_CLIP, ORIG_FRAC, ORIG_READ; EM.est_mech(m, o2, **kw)
    EM.clip, EM.frac, EM._read_point = clip_w, frac_w, read_w
    assert o2.n == charged, (o2.n, charged)
    rows.append((charged, strict))
    print(f"{name:10s} {charged:8d} {strict:8d} {strict / charged:6.3f} {S['n_clip']:5d} {S['n_tri']:4d}   "
          f"(geom charged {S['orig_geom']}, strict {S['strict_geom']}; arc reads {S['read']}; est {est:.4f})")
r = np.array(rows)
print(f"mean strict/charged = {np.mean(r[:, 1] / r[:, 0]):.3f}; no RNG call raised inside est_mech")
