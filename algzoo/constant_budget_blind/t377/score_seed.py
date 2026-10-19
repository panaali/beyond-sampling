"""Run the unchanged shared scorer (algzoo/score.py) on a seeded AlgZoo model.

score.py only exposes --zoo H,T (default seed). This wrapper patches alg_zoo.zoo_2nd_argmax to fix `seed`
before score.py imports it; nothing else in score.py changes (same 200k held-out inputs, seed 20261016).
Usage: python score_seed.py <labels.py> <out.json> <H> <seed|example>
"""
import runpy
import sys
from pathlib import Path

import alg_zoo

REPO = Path(__file__).resolve().parents[3]
labels, out, H, seed = sys.argv[1:5]
argv = [str(REPO / "algzoo/score.py"), labels, out]
if seed != "example":
    _orig = alg_zoo.zoo_2nd_argmax
    alg_zoo.zoo_2nd_argmax = lambda h, t, **kw: _orig(h, t, seed=int(seed))
    argv += ["--zoo", f"{H},10"]
sys.argv = argv
runpy.run_path(argv[0], run_name="__main__")
