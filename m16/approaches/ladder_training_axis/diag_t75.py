"""T101 diagnosis of T75 error (exact truth used only to attribute error, not by any estimator).
For zoo seed s at rung k, on n copies of a DEV noise seed (default 0; never the 20-23 test panel), compare:
  frozen  = est75.RECIPE (eps 0.1 at both steps)
  s3only  = eps (0, 0.1): every cutting step-2 gate resolved, only step-3 gates frozen
  s2only  = eps (0.1, 0): only step-2 gates frozen
  exact0  = eps (0, 0): every cutting gate resolved (should equal exact.py)
  margin  = rule 'margin' (est75's margin-aware step-3 impact, from T85/R3; NOT the frozen recipe)
and report MSE ratio vs sampling at each variant's own ops.
usage: python diag_t75.py k s [n=64] [noise_seed=0] [variants]"""
import sys

import numpy as np

import lt
from zoo43 import noisy_copies

VARIANTS = {"frozen": dict(K=(4, 4), rule="adaptive", eps=0.1), "s3only": dict(K=(4, 4), rule="adaptive", eps=(0.0, 0.1)),
            "s2only": dict(K=(4, 4), rule="adaptive", eps=(0.1, 0.0)), "exact0": dict(K=(4, 4), rule="adaptive", eps=(0.0, 0.0)),
            "margin": dict(K=(4, 4), rule="margin", eps=0.1)}


def run(k, s, n=64, ns=0, names=None):
    base = lt.zoo(k, s); ms = noisy_copies(base, n, lt.SIGMA, ns)
    tr = np.array([lt.truth(m)[0] for m in ms]); pq = np.mean(tr * (1 - tr))
    out = {}
    for nm in (names or VARIANTS):
        e, o = [], []
        for m in ms:
            r, ops, _ = lt.t75(m, VARIANTS[nm]); e.append(r); o.append(ops)
        e = np.array(e) - tr; om = np.mean(o)
        out[nm] = dict(ratio=float(np.mean(e ** 2) / (pq / (om / lt.OPS_PER_SAMPLE))), rms=float(np.sqrt(np.mean(e ** 2))),
                       bias=float(e.mean()), max_abs=float(np.abs(e).max()), ops=float(om), errs=e.tolist())
        print(f"k={k} s={s} dev ns={ns} n={n} {nm:7s} ratio {out[nm]['ratio']:.4f} rms {100*out[nm]['rms']:.2f}% bias {100*out[nm]['bias']:+.2f}% "
              f"max {100*out[nm]['max_abs']:.1f}% ops {om:.0f}", flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    run(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 64, int(a[3]) if len(a) > 3 else 0, a[4].split(",") if len(a) > 4 else None)
