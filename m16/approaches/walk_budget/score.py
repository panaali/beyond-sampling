"""T429 runner: frozen walk-quadrature recipes on DEV3 at a given budget, one deterministic run per model.

usage: python score.py RECIPE BUDGET OUT.jsonl [--walk py|nb] [--shard i/k]
  RECIPE: isg (T391, frozen/isg/estimator.py) or t10 (T10 recipe, frozen/estimator.py)
Rows already in OUT.jsonl are skipped (resume). Two execution-only swaps, both checked by gate.py:
  * C.sobol returns a lazy rule that generates the requested slice of the same scrambled-Sobol sequence on demand
    (the frozen code materialises 2^ceil(log2(B/2+64)) x 10 points per stratum: 3.4 GB at 1e7 FPE);
  * --walk nb: T393's compiled walk (repro mode) instead of the numpy walk.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
FZ = os.path.join(HERE, "frozen")
DATA = os.path.join(HERE, "..", "..", "..", "data", "dev3")
SHA = {  # sha256[:12], as pre-registered by T391 (e3a83b95) / T393 (c7145299) / T342
    "isg/estimator.py": "0bb493e6b480", "pos/table.npz": "e772b13f333a", "estimator.py": "fb7902e12240",
    "walk.py": "a38258f586b4", "hom.py": "a91f019fca21", "core.py": "48dbba49ab13", "walk_nb.py": "bbc6cc992c7c",
}
DATA_SHA = {"weights.npz": "62c952037e8c", "truths.json": "abcb7d6d264a"}


def _h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


for f, s in SHA.items():
    assert _h(os.path.join(FZ, f)) == s, f
for f, s in DATA_SHA.items():
    assert _h(os.path.join(DATA, f)) == s, f


class LazyRule:
    """rules[lo:hi] of qmc.Sobol(nd, scramble=True, seed).random_base2(m)[:n], generated on demand."""

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


def load(recipe, walk="nb", lazy=True):
    sys.path.insert(0, FZ)
    import core as C
    import walk as W
    if lazy:
        C.sobol = LazyRule
    if walk == "nb":
        import walk_nb as WN
        WN.MODE["fast"] = False; WN.MODE["fm"] = False
        W.walk = WN.walk
    path = os.path.join(FZ, "isg", "estimator.py") if recipe == "isg" else os.path.join(FZ, "estimator.py")
    spec = importlib.util.spec_from_file_location(f"est_{recipe}", path)
    E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    return E


def models():
    W = np.load(os.path.join(DATA, "weights.npz"))
    return {str(n): (W["Wi"][i], W["Wh"][i], W["Wo"][i]) for i, n in enumerate(W["names"])}


if __name__ == "__main__":
    recipe, budget, out = sys.argv[1], int(float(sys.argv[2])), sys.argv[3]
    args = sys.argv[4:]
    walk = args[args.index("--walk") + 1] if "--walk" in args else "nb"
    i, k = map(int, args[args.index("--shard") + 1].split("/")) if "--shard" in args else (0, 1)
    E = load(recipe, walk)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["model"] for l in open(out) if l.strip()}
    mods = models()
    for n, (mn, m) in enumerate(mods.items()):
        if n % k != i or mn in done:
            continue
        t0, w0 = time.process_time(), time.time()
        e, c = E.est(m, budget, None)
        row = dict(model=mn, recipe=recipe, budget=budget, est=e, cost=c, cpu=time.process_time() - t0,
                   wall=time.time() - w0, walk=walk)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(mn, f"{e:.9f} {c:.0f} {row['cpu']:.1f}s", flush=True)
