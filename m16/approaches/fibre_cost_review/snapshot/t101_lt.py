"""T101 shared helpers (training-amount ladder for M_4,3). Imports other lanes read-only; edits nothing.

* zoo(k, s): weights of alg_zoo.zoo_2nd_argmax(4, 3, n_seqs=2**k, seed=s) as float64 (Wi (4,), Wh (4,4), Wo (3,4)).
* panel(base): the frozen ARC-protocol noisy panel: absolute sigma 0.05 on every weight, noise seeds 20-23 x 256
  copies (= understand_m43.zoo43.noisy_copies, the E37 fresh-seed convention; 1024 copies per base model).
  The SAME noise draws are added to every k (common random numbers across the ladder).
* truth(m): exact accuracy, arc_m43_repro/exact.py (also its own op count, author convention).
* regions(m): number of linear regions (distinct (sign x0, g2, g3) patterns), review/understand_ladder/rtruth.exact.
* t75(m): frozen T75 recipe understand_m43/est75.RECIPE; ops = est75's strict charge (T74/R3 convention).
* f2l3(m): fibre recipe F2 with look-ahead L (default 3) built from review/understand_ladder/fibre_strict.py (strict
  recount: ops = n + x) with R4's one-line id()-reuse cache fix (the fibre_fix.py line) patched in at import time.
* OPS_PER_SAMPLE = 72 (ARC). Sampling MSE at equal ops = p(1-p) / (ops / 72).
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..")
ROOT = os.path.join(HERE, "..", "..", "..")
for d in (os.path.join(APP, "arc_m43_repro"), os.path.join(APP, "understand_m43"), os.path.join(APP, "understand_ladder"),
          os.path.join(ROOT, "review", "understand_ladder")):
    if d not in sys.path:
        sys.path.append(d)

import exact as EX  # noqa: E402
import est75_e37 as E75  # noqa: E402  (pinned copy of understand_m43/est75.py @ d406c17; bit-identical to live RECIPE on 140 models)
from m43 import Ops, OPS_PER_SAMPLE  # noqa: E402
from zoo43 import noisy_copies  # noqa: E402
import rtruth as RT  # noqa: E402

NOISE_SEEDS = (20, 21, 22, 23)
N_PER_SEED = 256
SIGMA = 0.05


def _fixed_strict_fibre():
    src_path = os.path.join(ROOT, "review", "understand_ladder", "fibre_strict.py")
    src = open(src_path).read()
    anchor = "    stc = State(model, x, P, U, nn, ops)\n"
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + "    cache.setdefault(\"_alive\", []).append(stc)   # R4 fix (fibre_fix.py)\n")
    mod = types.ModuleType("fibre_strict_fixed"); mod.__file__ = src_path
    exec(compile(src, src_path + " [+R4 fix]", "exec"), mod.__dict__)
    return mod


FIB = _fixed_strict_fibre()


def zoo(k, s):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(4, 3, n_seqs=2 ** k, seed=s)
    sd = {a: v.detach().double().numpy().copy() for a, v in m.state_dict().items()}
    return sd["rnn.weight_ih_l0"][:, 0], sd["rnn.weight_hh_l0"], sd["linear.weight"]


def panel(base, n_per_seed=N_PER_SEED):
    return [m for ns in NOISE_SEEDS for m in noisy_copies(base, n_per_seed, SIGMA, ns)]


def truth(m):
    o = Ops(); a = EX.exact_acc(m, o)
    return a, o.n


def regions(m):
    return RT.exact(m)[1]


def t75(m, recipe=None, mod=None):
    """mod=None: the pinned E37 copy (frozen RECIPE); mod=live_est75() for later rules (margin, T85/T112)."""
    M = mod or E75
    M.STATS.update(cut=0, resolved=0, cells=0)
    o = Ops(); r = M.est75(m, o, **(recipe or E75.RECIPE))
    return r, o.n, dict(M.STATS)


def live_est75():
    import est75
    return est75


def f2l(m, look=3):
    FIB.LOOK = look
    o = FIB.Ops(); inf = {}
    e = FIB.est_fibre2(m, o, inf)
    return e, o.n + getattr(o, "x", 0), inf.get("traces")
