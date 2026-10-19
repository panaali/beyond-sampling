"""T442 runner: the frozen T10 walk-quadrature recipe (T429 @ 79ecae63) with ONLY the line set (node rule) changed.

usage: python run.py FAMILY SEED BUDGET OUT.jsonl [--shard i/k]
  FAMILY: sobol    scrambled Sobol (T429's lazy rule), rule seed SEED (frozen recipe: SEED = 31415)
          lat      Kuo/CKN embedded rank-1 lattice (radical-inverse order), random shift from SEED
          latb     same lattice and shift, then the baker's (tent) periodising transform x -> 1 - |2x - 1|
          lat0     same lattice, no shift, no transform (fixed rule; SEED ignored)
          halton   scrambled Halton (scipy), seed SEED
Every family is called exactly as the frozen code calls C.sobol(T, nmax, RULE_SEED + 101 j) for stratum j, with
RULE_SEED = SEED, and is consumed as prefixes rule[lo:hi]; everything else (strata, face maps, walk, pilot, Neyman
allocation, cost) is T429's code, hash-asserted in t429_score.py.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
assert hashlib.sha256(open(os.path.join(HERE, "t429_score.py"), "rb").read()).hexdigest()[:12] == "afcb43325500"
sys.path.insert(0, HERE)
import t429_score as S  # noqa: E402  (asserts the seven frozen-file hashes and the DEV3 data hashes)

M_LAT = 20
Z_KUO = np.array([int(l) for l in open(os.path.join(HERE, "kuo_lattice-33002-1024-1048576_first32.txt"))
                  if l.strip() and not l.startswith("#")], dtype=np.uint64)


def _rev(k, m):
    k = k.astype(np.uint64); r = np.zeros_like(k)
    for _ in range(m):
        r = (r << np.uint64(1)) | (k & np.uint64(1)); k >>= np.uint64(1)
    return r


class LatRule:
    """Embedded rank-1 lattice sequence x_k = frac(phi_2(k) z + shift), k < 2^20; optional baker's transform."""

    def __init__(self, nd, n, seed, shift=True, baker=False):
        assert nd <= len(Z_KUO)
        self.nd, self.n, self.z, self.baker = nd, n, Z_KUO[:nd], baker
        self.shift = np.random.default_rng(seed).random(nd) if shift else np.zeros(nd)

    def __getitem__(self, s):
        lo, hi = s.start or 0, min(s.stop if s.stop is not None else self.n, self.n)
        if hi <= lo:
            return np.zeros((0, self.nd))
        assert hi <= 1 << M_LAT, "lattice sequence exhausted (2^20 points per stratum)"
        k = _rev(np.arange(lo, hi), M_LAT)
        x = ((k[:, None] * self.z[None, :]) & np.uint64((1 << M_LAT) - 1)).astype(np.float64) / (1 << M_LAT)
        x = (x + self.shift) % 1.0
        if self.baker:
            x = 1.0 - np.abs(2.0 * x - 1.0)
        return np.minimum(x, 1.0 - 2.0 ** -53)


class HaltonRule(S.LazyRule):
    def __getitem__(self, s):
        lo, hi = s.start or 0, min(s.stop if s.stop is not None else self.n, self.n)
        if hi <= lo:
            return np.zeros((0, self.nd))
        if self.eng is None or self.pos != lo:
            self.eng = qmc.Halton(self.nd, scramble=True, seed=self.seed)
            if lo:
                self.eng.fast_forward(lo)
        self.pos = hi
        return self.eng.random(hi - lo)


FACT = {
    "sobol": None,  # T429's LazyRule (installed by S.load)
    "lat": lambda nd, n, s: LatRule(nd, n, s),
    "latb": lambda nd, n, s: LatRule(nd, n, s, baker=True),
    "lat0": lambda nd, n, s: LatRule(nd, n, s, shift=False),
    "halton": lambda nd, n, s: HaltonRule(nd, n, s),
}


def load(family):
    E = S.load("t10", "nb")
    if FACT[family] is not None:
        E.C.sobol = FACT[family]
    return E


if __name__ == "__main__":
    fam, seed, budget, out = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3])), sys.argv[4]
    args = sys.argv[5:]
    i, k = map(int, args[args.index("--shard") + 1].split("/")) if "--shard" in args else (0, 1)
    E = load(fam)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["model"] for l in open(out) if l.strip()}
    for n, (mn, m) in enumerate(S.models().items()):
        if n % k != i or mn in done:
            continue
        t0, w0 = time.process_time(), time.time()
        e, c = E.est(m, budget, None, rule_seed=seed)
        row = dict(model=mn, family=fam, seed=seed, budget=budget, est=e, cost=c, cpu=time.process_time() - t0,
                   wall=time.time() - w0)
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(mn, f"{e:.9f} {c:.0f} {row['cpu']:.1f}s", flush=True)
