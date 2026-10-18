"""T335 gate: (a) estimator_f imports and runs with numpy's random generators patched to raise, loading none of the
development modules or scipy.stats; (b) on all 80 DEV2, estimates bitwise equal to the T330 tier-C C1_all run
(results_tierc.json) and to screen_c (mode C1, all lines) live; FLOPs equal to that run minus exactly the 3 x (H^2 + 2C^2H)
FLOPs that screen_c charged for tier-B tables it never used in C1 mode (cont_f charges as executed). -> results_f.json"""
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
sys.path.insert(0, sys.argv[1]); import estimator_f, cont_f
bad = [m for m in ("walk", "r1w2", "r1w", "cont", "cont_c", "cont_d", "cont_e", "r0", "v1b2", "screen_c", "scipy.stats") if m in sys.modules]
print("loaded:", bad); assert not bad
Wi = np.sin(np.arange(16.0)); Wh = 0.3 * np.cos(np.add.outer(np.arange(16.0), 2 * np.arange(16.0)))
Wo = np.sin(np.add.outer(3 * np.arange(10.0), np.arange(16.0)))
print("est call (no random draw):", estimator_f.est((Wi, Wh, Wo)))
'''
pr = subprocess.run([sys.executable, "-c", probe, HERE], capture_output=True, text=True)
print(pr.stdout.strip(), pr.stderr.strip()[-300:]); purity = pr.returncode == 0

import numpy as np  # noqa: E402
import estimator_f  # noqa: E402
import screen_c as S  # noqa: E402
from cont_e import H, C  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
import r0  # noqa: E402

OVER = 3 * (H * H + 2 * C * C * H)
ref = json.load(open(os.path.join(HERE, "results_tierc.json")))["rows"]
rows = {}
for nm, m in r0.dev2():
    ef, ff = estimator_f.est(m); es, fs = S.est(m, lines=("nb-23", "nb-12", "b-23"), mode="C1")
    rr = ref[nm]["C1_all"]
    rows[nm] = dict(est=ef, C=ff / 5948, est_eq_run=bool(ef == rr["est"]), est_eq_live=bool(ef == es),
                    flops_eq_live_minus_over=bool(ff == fs - OVER), C_run=rr["C"])
s = dict(purity=purity, est_bitwise_run=all(r["est_eq_run"] for r in rows.values()),
         est_bitwise_live=all(r["est_eq_live"] for r in rows.values()),
         flops_exact_live_minus_unused=all(r["flops_eq_live_minus_over"] for r in rows.values()),
         C_f=float(np.mean([r["C"] for r in rows.values()])), C_run=float(np.mean([r["C_run"] for r in rows.values()])),
         unused_charge_FPE=OVER / 5948)
print(json.dumps(s, indent=1))
json.dump(dict(summary=s, rows=rows), open(os.path.join(HERE, "results_f.json"), "w"), indent=1)
