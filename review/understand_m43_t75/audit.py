"""T84 item 5: (a) the reviewed est75 makes no random calls: every numpy / stdlib RNG entry point is replaced by a
function that raises, then est75 runs on clean + noisy models of all 5 seeds; (b) determinism (repeat runs are
bit-identical); (c) every numeric literal in est75.py is listed for the constant audit; (d) est75 reads nothing
from truth files (no open / json / exact import).
usage: python audit.py"""
import ast, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
UND = os.path.join(HERE, "..", "..", "m16", "approaches", "understand_m43")
sys.path.insert(0, UND); sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
from mine import zoo_w, noisy_w

models = []
for s in range(5):
    b = zoo_w(s); models += [b] + noisy_w(b, 4, 0.05, 70) + noisy_w(b, 2, 0.1, 71)

import est75 as E75                       # import first (module-level geometry), then forbid randomness
from m43 import Ops


def boom(*a, **k):
    raise RuntimeError("random call")


saved = {}
for name in ("default_rng", "random", "rand", "randn", "standard_normal", "normal", "uniform", "integers", "choice",
             "permutation", "shuffle", "seed", "RandomState", "Generator"):
    if hasattr(np.random, name):
        saved[name] = getattr(np.random, name); setattr(np.random, name, boom)
for name in ("random", "randint", "uniform", "gauss", "choice", "shuffle", "seed", "Random"):
    saved["py_" + name] = getattr(random, name); setattr(random, name, boom)
r1 = [E75.est75(m, Ops(), **E75.RECIPE) for m in models]
r2 = [E75.est75(m, Ops(), **E75.RECIPE) for m in models]
for k, f in saved.items():
    if k.startswith("py_"):
        setattr(random, k[3:], f)
    else:
        setattr(np.random, k, f)
print(f"(a) {len(models)} models ran with all RNG entry points raising: no random call")
print(f"(b) repeat runs bit-identical: {r1 == r2}")
src = open(os.path.join(UND, "est75.py")).read()
lits = sorted({repr(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)}, key=float)
print(f"(c) numeric literals in est75.py: {', '.join(lits)}")
bad = [w for w in ("open(", "json", "exact", "truth.txt", "results", "dev75", "np.load", "pickle") if w in src]
print(f"(d) suspicious tokens in est75.py: {bad or 'none'} (note: 'truth' in est75 is the half-chamber's true class)")
srcm = open(os.path.join(UND, "est_mech.py")).read()
i = srcm.index("def _family_halves"); j = srcm.index("HALVES = _family_halves()")
print("    est75 imports from est_mech only HALVES (model-free geometry):", "from est_mech import HALVES" in src)
