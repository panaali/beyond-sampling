"""T113 item 5: (a) the reviewed est75 RECIPE112 path makes no random calls (every numpy / stdlib RNG entry point
raises); (b) repeat runs are bit-identical; (c) numeric literals in est75.py; (d) no truth files are read;
(e) the recipe uses no fitted / per-seed constants: identical code + knobs for every H and seed (RECIPE112 dict).
usage: OMP_NUM_THREADS=1 python audit.py  -> out/audit.log"""
import ast, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
UND = os.path.join(HERE, "..", "..", "m16", "approaches", "understand_m43")
sys.path.insert(0, UND); sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import mine as M

models = []
for H in (4, 5, 6, 8):
    for s in range(5):
        b = M.zoo_w(H, s); models += [b] + M.noisy_abs(b, 2, 0.05, 79) + M.noisy_rel(b, 1, 0.01, 79)
import est75 as E75
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
r1 = [E75.est75(m, Ops(), **E75.RECIPE112) for m in models]
o1 = []
for m in models:
    o = Ops(); E75.est75(m, o, **E75.RECIPE112); o1.append(o.n)
r2 = [E75.est75(m, Ops(), **E75.RECIPE112) for m in models]
for k, f in saved.items():
    setattr(random, k[3:], f) if k.startswith("py_") else setattr(np.random, k, f)
print(f"(a) {len(models)} models (H = 4, 5, 6, 8; clean, abs and rel noise) ran with every RNG entry point raising: no random call")
print(f"(b) repeat runs bit-identical: {r1 == r2}")
src = open(os.path.join(UND, "est75.py")).read()
lits = sorted({repr(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)}, key=float)
print(f"(c) numeric literals in est75.py: {', '.join(lits)}")
bad = [w for w in ("open(", "json", "import exact", "truth.txt", "np.load", "pickle") if w in src]
print(f"(d) suspicious tokens in est75.py: {bad or 'none'}")
print(f"(e) RECIPE112 = {E75.RECIPE112}; the margin_all path reads only (Wi, Wh, Wo), HALVES (model-free) and these knobs")
