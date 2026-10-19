"""python run_list.py T name...  -> exact accuracy of each named leaderboard model via estimator.est (logs est_info.jsonl)."""
import sys
import time

sys.path.append("../..")
import leaderboard as L  # noqa: E402
import estimator  # noqa: E402

T = int(sys.argv[1])
ms = L.models(16, T)
for name in sys.argv[2:]:
    t0 = time.time()
    a, c = estimator.est(ms[name], 0, None)
    print(f"{name} acc {a:.10f} fpe {c:.4g} wall {time.time() - t0:.0f}s", flush=True)
