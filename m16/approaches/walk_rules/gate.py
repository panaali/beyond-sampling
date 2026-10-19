"""T442 gate (run before any scoring): (1) family 'sobol' with seed 31415 reproduces T429's saved T10 rows bitwise
(5 models at 256k); (2) lattice rule checks: prefix consistency, 2^m prefixes of the unshifted rule are full lattices,
points in [0, 1); (3) Halton prefix consistency."""
import json
import os

import numpy as np

import run as R

HERE = os.path.dirname(os.path.abspath(__file__))
saved = {r["model"]: r["est"] for r in map(json.loads, open(os.path.join(HERE, "..", "walk_budget", "runs",
                                                                         "t10_256000.jsonl")))}
E = R.load("sobol"); mods = R.S.models()
d = [E.est(mods[n], 256000, None, rule_seed=31415)[0] - saved[n] for n in list(mods)[:5]]
print("1. sobol 31415 vs T429 t10_256000 (5 models):", d, "PASS" if all(x == 0 for x in d) else "FAIL")
ok = True
for fam in ("lat", "latb", "lat0", "halton"):
    r1, r2 = R.FACT[fam](10, 1 << 21, 777), R.FACT[fam](10, 1 << 21, 777)
    a = r1[0:5000]; b = np.r_[r2[0:1234], r2[1234:5000]]
    ok &= np.array_equal(a, b) and a.min() >= 0 and a.max() < 1
r0 = R.FACT["lat0"](10, 1 << 21, 0)
for m in (4, 10, 16):
    x = r0[0:1 << m]
    ok &= all(np.array_equal(np.sort(x[:, j]), np.arange(1 << m) / (1 << m)) for j in range(10))
print("2-3. lattice / halton rule checks:", "PASS" if ok else "FAIL")
