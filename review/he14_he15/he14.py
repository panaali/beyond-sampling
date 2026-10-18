"""T280 H-E14: region counts k = 2..4 for target, frob0-4, row0-4 and the 5 activity-matched mu draws.
usage: python he14.py -> results/he14.json"""
import json
import os
import subprocess
import sys

import numpy as np

import models as M

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
pool = {f"mu{s}": M.activity(M.get(f"mu{s}")) for s in range(2900, 3000)}
a0 = M.activity(M.target())
mu_sel = sorted(pool, key=lambda k: abs(pool[k] - a0))[:5]
print("target activity", a0, "mu pool median", np.median(list(pool.values())), "selected",
      {k: round(pool[k], 3) for k in mu_sel}, flush=True)
names = ["target"] + [f"frob{i}" for i in range(5)] + [f"row{i}" for i in range(5)] + mu_sel
out = {"activity_target": a0, "mu_pool_activity": pool, "mu_selected": mu_sel, "models": {}}
for nm in names:
    r = {"activity": M.activity(M.get(nm))}
    for k in (2, 3, 4):
        line = subprocess.run([sys.executable, "dd.py", "prefix", nm, str(k)], capture_output=True, text=True,
                              cwd=HERE).stdout.strip().splitlines()[-1]
        d = json.loads(line); r[f"k{k}"] = d["regions"]; r[f"cells_k{k}"] = d["cells"]
    out["models"][nm] = r; print(nm, r, flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "he14.json"), "w"), indent=1)
