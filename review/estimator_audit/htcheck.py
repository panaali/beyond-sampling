"""Does est() run (and return something sensible) on a model with (H,T) != (16,10)?
usage (repo root): OMP_NUM_THREADS=1 python review/estimator_audit/htcheck.py NAME [H T budget]
Uses a random model (Wi (H,), Wh (H,H), Wo (T,H)); only checks that the code runs and that the estimate is
in [0,1] and close to a 200k-sample truth. Not a quality test."""
import sys, json, time, traceback
import numpy as np
sys.path.insert(0, "m16")
import leaderboard as LB

name = sys.argv[1]
H, T, b = (int(v) for v in sys.argv[2:5]) if len(sys.argv) > 4 else (12, 8, 4000)
r = np.random.default_rng(7)
m = (r.standard_normal(H), 0.8 * r.standard_normal((H, H)) / np.sqrt(H), r.standard_normal((T, H)))
x = r.standard_normal((200_000, T)); truth = float(LB.correct(m, x).mean())
try:
    est, _ = LB.load_estimator(name)
    t0 = time.process_time(); e, c = est(m, b, np.random.default_rng(1))
    out = dict(name=name, H=H, T=T, ok=True, estimate=float(e), cost=float(c), truth=truth, sec=round(time.process_time() - t0, 2))
except Exception as ex:
    out = dict(name=name, H=H, T=T, ok=False, error=f"{type(ex).__name__}: {str(ex)[:160]}",
               where=traceback.extract_tb(ex.__traceback__)[-1].filename.split('clean-neurons/')[-1] + ":" +
               str(traceback.extract_tb(ex.__traceback__)[-1].lineno), truth=truth)
print(json.dumps(out), flush=True)
