"""Wall-clock sanity check: est(orig, budget, rng) vs plain sampling at the estimator's reported cost.
usage (repo root, venv active): OMP_NUM_THREADS=1 python review/estimator_audit/timecheck.py NAME [budget]
Prints: name, reported cost, est seconds, sampling seconds at that cost, and
ratio = est_time / sampling_time(cost); ratio_warm uses a second call (lazy loads and caches already done)."""
import sys, time, json
import numpy as np
sys.path.insert(0, "m16")
import leaderboard as LB
from arc_eval import Wi0, Wh0, Wo0, est_sampling

name = sys.argv[1]; budget = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
m = (Wi0, Wh0, Wo0)
est, attrs = LB.load_estimator(name)
t0 = time.process_time(); e, c = est(m, budget, np.random.default_rng(1)); t_est = time.process_time() - t0
t0 = time.process_time(); est(m, budget, np.random.default_rng(2)); t_warm = time.process_time() - t0   # 2nd call: lazy loads/caches done
n = max(1, int(round(c)))
ts = []
for r in range(5):
    t0 = time.process_time(); est_sampling(m, n, np.random.default_rng(r)); ts.append(time.process_time() - t0)
t_s = float(np.median(ts))
print(json.dumps(dict(name=name, budget=budget, estimate=float(e), cost=float(c), t_est=round(t_est, 3), t_warm=round(t_warm, 3),
                      t_sampling_at_cost=round(t_s, 4), ratio=round(t_est / t_s, 1), ratio_warm=round(t_warm / t_s, 1))), flush=True)
