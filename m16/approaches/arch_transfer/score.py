"""M1: the frozen T10 walk-quadrature recipe (T429 frozen/estimator.py fb7902e12240) with the generic walk.

usage: python score.py TAG SEED BUDGET RULE [OUT.jsonl]     (RULE r -> rule_seed 31415 + 1000 r)
Execution-only changes to the frozen recipe, applied to its source text at import (hash checked first):
  * the model argument is a walkgen.Net (T = 10) and its walk is walkgen.walk (exact or numerical by net.mode);
  * C.sobol is T429's lazy rule (bitwise the same points, generated on demand; T429 gate).
radial = net.homog (Student-t radial integrand only for positively homogeneous nets; else the Gaussian window).
"""
import hashlib
import json
import os
import sys
import time
import types

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
FZ = os.path.join(HERE, "frozen")
sys.path.insert(0, HERE); sys.path.insert(0, FZ)
SHA = {"estimator.py": "fb7902e12240", "core.py": "48dbba49ab13", "hom.py": "a91f019fca21", "walk.py": "a38258f586b4"}
for f, s in SHA.items():
    assert hashlib.sha256(open(os.path.join(FZ, f), "rb").read()).hexdigest()[:12] == s, f

import core as C  # noqa: E402
import load  # noqa: E402
import walkgen as G  # noqa: E402


class LazyRule:
    """rules[lo:hi] of qmc.Sobol(nd, scramble=True, seed).random_base2(m)[:n], generated on demand (T429)."""

    def __init__(self, nd, n, seed):
        self.nd, self.n, self.seed, self.eng, self.pos = nd, n, seed, None, 0

    def __getitem__(self, s):
        lo, hi = s.start or 0, min(s.stop if s.stop is not None else self.n, self.n)
        if hi <= lo:
            return np.zeros((0, self.nd))
        if self.eng is None or self.pos != lo:
            self.eng = qmc.Sobol(self.nd, scramble=True, seed=self.seed)
            if lo:
                self.eng.fast_forward(lo)
        self.pos = hi
        return self.eng.random(hi - lo)


C.sobol = lambda nd, n, seed: LazyRule(nd, n, seed)

src = open(os.path.join(FZ, "estimator.py")).read()
for old, new in [
    ("import walk as W  # noqa: E402", "import walkgen as W  # noqa: E402"),
    ("    model = tuple(np.asarray(w, dtype=np.float64) for w in model)\n    T = model[2].shape[0]", "    T = 10"),
]:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
E = types.ModuleType("estimator_gen")
E.__file__ = os.path.join(FZ, "estimator.py")
exec(compile(src, "estimator_gen", "exec"), E.__dict__)


def run(tag, seed, budget, rule):
    net = G.Net(*load.load(tag, seed))
    t0 = time.time()
    p, used = E.est(net, budget, rule_seed=E.RULE_SEED + 1000 * rule, radial=net.homog)
    return {"tag": tag, "seed": seed, "budget": budget, "rule": rule, "est": p, "cost": used,
            "mode": net.mode, "radial": net.homog, "fpe_flops": net.fpe, "wall": time.time() - t0}


if __name__ == "__main__":
    tag, seed, budget, rule = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3])), int(sys.argv[4])
    out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(HERE, "runs", f"m1_{tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        for l in open(out):
            r = json.loads(l)
            if (r["seed"], r["budget"], r["rule"]) == (seed, budget, rule):
                print("done already"); sys.exit(0)
    r = run(tag, seed, budget, rule)
    with open(out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps(r))
