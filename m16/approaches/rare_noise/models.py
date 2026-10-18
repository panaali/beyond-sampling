"""T371 model panel: 4 clean bases x 4 noise levels x 3 copies, by m16/arc_eval.py's noise protocol.
Noise: W + sigma * RMS(W) * N(0, 1) per matrix, rng = default_rng(0), copies drawn in order (arc_eval.noisy_models).
For orig we call arc_eval.noisy_models itself; for the DEV3 bases we apply the same formula (asserted to reproduce
arc_eval bitwise on orig). Same seed at every level, so copy j at each sigma lies on one noise ray (as arc_eval does).
DEV3 weights: claude/t286-cheap-frontier rare_events/ext/dev3/weights.npz (sha256 asserted), env T371_DEV3."""
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "m16"))
import arc_eval  # noqa: E402

BASES = ("orig", "dev3_s526", "dev3_s528", "dev3_s530")
SIGMAS = (0.001, 0.005, 0.02, 0.05)
NCOPY = 3
DEV3_SHA = "62c952037e8c9ec214ee96b3dc43923a9e19cb5f137dbf16b7f6308d721591f9"   # as cited in rare_events README (T342 weights)


def _noisy(base, k, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(W + sigma * np.sqrt((W ** 2).mean()) * rng.standard_normal(W.shape) for W in base) for _ in range(k)]


def bases():
    out = {"orig": (arc_eval.Wi0, arc_eval.Wh0, arc_eval.Wo0)}
    fn = os.environ["T371_DEV3"]; assert hashlib.sha256(open(fn, "rb").read()).hexdigest() == DEV3_SHA
    z = np.load(fn); names = list(map(str, z["names"]))
    for nm in BASES[1:]:
        i = names.index(nm); out[nm] = tuple(np.asarray(z[k][i], np.float64) for k in ("Wi", "Wh", "Wo"))
    return out


def panel():
    """name -> (Wi, Wh, Wo) float64. Names: '<base>' (clean) and '<base>@<sigma>#<j>'."""
    B = bases(); out = {}
    ref = arc_eval.noisy_models(NCOPY, SIGMAS[0])
    mine = _noisy(B["orig"], NCOPY, SIGMAS[0])
    assert all(np.array_equal(a, b) for r, m in zip(ref, mine) for a, b in zip(r, m)), "noise replica != arc_eval"
    for nm in BASES:
        out[nm] = B[nm]
        for s in SIGMAS:
            cps = arc_eval.noisy_models(NCOPY, s) if nm == "orig" else _noisy(B[nm], NCOPY, s)
            for j, m in enumerate(cps):
                out[f"{nm}@{s:g}#{j}"] = tuple(np.asarray(w, np.float64) for w in m)
    return out
