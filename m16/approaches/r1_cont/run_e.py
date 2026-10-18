"""T317e gate: (a) importing estimator_e runs no random draw and loads none of walk / r1w2 / cont / cont_c / cont_d / r0
(checked in a fresh subprocess with numpy's random generators patched to raise); (b) estimates and FLOPs bitwise
identical to T317d (cont_d.est) on all 80 DEV2. -> results_e.json"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
probe = r'''
import sys, numpy as np
def boom(*a, **k): raise RuntimeError("random draw")
np.random.default_rng = boom; np.random.seed = boom; np.random.standard_normal = boom; np.random.random = boom
np.random.RandomState = boom; np.random.Generator = boom
sys.path.insert(0, sys.argv[1]); import estimator_e, cont_e
bad = [m for m in ("walk", "r1w2", "r1w", "cont", "cont_c", "cont_d", "r0", "v1b2", "scipy.stats") if m in sys.modules]
print("loaded:", bad); assert not bad
Wi = np.sin(np.arange(16.0)); Wh = 0.3 * np.cos(np.add.outer(np.arange(16.0), 2 * np.arange(16.0)))
Wo = np.sin(np.add.outer(3 * np.arange(10.0), np.arange(16.0)))
print("est call (no random draw):", estimator_e.est((Wi, Wh, Wo)))
'''
pr = subprocess.run([sys.executable, "-c", probe, HERE], capture_output=True, text=True)
print(pr.stdout.strip(), pr.stderr.strip()[-300:]); purity = pr.returncode == 0

import numpy as np  # noqa: E402
import cont_d  # noqa: E402
import estimator_e  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
import r0  # noqa: E402

rows = {}
for nm, m in r0.dev2():
    ad, fd = cont_d.est(m); ae, fe = estimator_e.est(m)
    rows[nm] = dict(est_d=ad, est_e=ae, flops_d=fd, flops_e=fe, bitwise=bool(ad == ae and fd == fe))
s = dict(purity=purity, all_bitwise=all(r["bitwise"] for r in rows.values()),
         max_abs_diff=max(abs(r["est_d"] - r["est_e"]) for r in rows.values()),
         max_abs_dflops=max(abs(r["flops_d"] - r["flops_e"]) for r in rows.values()),
         C_e=float(np.mean([r["flops_e"] for r in rows.values()])) / 5948)
print(json.dumps(s, indent=1))
json.dump(dict(summary=s, rows=rows), open(os.path.join(HERE, "results_e.json"), "w"), indent=1)
