"""T429 reproduction gate (run before any new-budget scoring).
1. LazyRule slices == random_base2 rule bitwise (several seeds, sizes, slice patterns).
2. Lazy rule + numpy walk: est bitwise equal to the saved T391 / T10-recipe DEV3 rows (first 5 models, 16k).
3. Lazy rule + compiled walk: est and cost within 1e-12 relative of the saved rows, all 100 models, 16k and 64k,
   both recipes. Saved rows: minimal_pair dev3_scores_isg.json (e3a83b95) and dev3_scores.json (a4097a95)."""
import json
import subprocess
import sys

import numpy as np
from scipy.stats import qmc

import score as S

ref = {"isg": json.loads(subprocess.check_output(
           ["git", "show", "e3a83b95:m16/approaches/minimal_pair/dev3_scores_isg.json"], cwd=S.HERE)),
       "t10": json.loads(subprocess.check_output(
           ["git", "show", "a4097a95:m16/approaches/minimal_pair/dev3_scores.json"], cwd=S.HERE))}
key = {"isg": "isg|{}", "t10": "minimal_pair|{}"}

for seed in (31415, 14142 + 202, 7):
    full = qmc.Sobol(10, scramble=True, seed=seed).random_base2(16)
    L = S.LazyRule(10, 1 << 16, seed)
    got = np.concatenate([L[0:16], L[16:48], L[48:1000], L[1000:40000], L[40000:1 << 16]])
    assert np.array_equal(got, full), seed
    assert np.array_equal(S.LazyRule(10, 1 << 16, seed)[777:5000], full[777:5000])
print("1. lazy rule bitwise: PASS", flush=True)

mods = S.models()
worst = 0.0
for recipe in ("isg", "t10"):
    rows = {r["model"]: r for r in ref[recipe][key[recipe].format(16000)]}
    E = S.load(recipe, walk="py")
    for mn in list(mods)[:5]:
        e, c = E.est(mods[mn], 16000, None)
        assert e == rows[mn]["est"] and c == rows[mn]["cost"], (recipe, mn, e, rows[mn]["est"])
    sys.path.remove(S.FZ)
    for m in ("core", "walk", "hom", "estimator", "walk_nb"):
        sys.modules.pop(m, None)
print("2. lazy rule + numpy walk bitwise (5 models x 2 recipes, 16k): PASS", flush=True)

for recipe in ("isg", "t10"):
    E = S.load(recipe, walk="nb")
    for b in (16000, 64000):
        rows = {r["model"]: r for r in ref[recipe][key[recipe].format(b)]}
        for mn, m in mods.items():
            e, c = E.est(m, b, None)
            worst = max(worst, abs(e - rows[mn]["est"]) / rows[mn]["est"], abs(c - rows[mn]["cost"]) / rows[mn]["cost"])
        print(recipe, b, "worst rel so far", worst, flush=True)
    sys.path.remove(S.FZ)
    for m in ("core", "walk", "hom", "estimator", "walk_nb"):
        sys.modules.pop(m, None)
print("3. compiled walk, 100 models x 2 budgets x 2 recipes: worst rel", worst, "PASS" if worst <= 1e-12 else "FAIL")
