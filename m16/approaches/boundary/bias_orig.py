import sys, json
import numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16"); sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/boundary")
from common import Wi, Wh, Wo
from estimator import est_boundary
M = (Wi, Wh, Wo); truth = 0.95317945
b, reps = int(sys.argv[1]), int(sys.argv[2])
e = []; u = []
for r in range(reps):
    a, uu = est_boundary(M, b, np.random.default_rng(10_000 + r)); e.append(a); u.append(uu)
e = np.array(e); d = e - truth
out = dict(budget=b, reps=reps, mean=float(e.mean()), bias=float(d.mean()), bias_se=float(d.std(ddof=1) / np.sqrt(reps)),
           rmse=float(np.sqrt((d ** 2).mean())), evals=float(np.mean(u)),
           sampling_rmse_at_evals=float(np.sqrt(truth * (1 - truth) / np.mean(u))))
print(out); json.dump(out, open(f"bias_orig_{b}.json", "w"), indent=1)
