"""Run the cone-chart B&B on noisy_models(3, 0.0025); compare its point estimates with sampling at equal FLOPs.
Ground truth (n=4M) from truth.py / truth.json."""
import json
import sys
import numpy as np

sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import noisy_models  # noqa: E402
import bb2  # noqa: E402

mf = float(sys.argv[1]) if len(sys.argv) > 1 else 3e10
truth = json.load(open("truth.json"))["0.0025"]
out = []
for i, m in enumerate(noisy_models(3, 0.0025)):
    log, _ = bb2.run(model=m, max_flops=mf, log_every=1.5, tag=f"m{i}")
    t = truth[i]
    for r in log:
        n = r["flops"] / 6000.0
        r.update(truth=t, sound=bool(r["L"] <= t <= r["U"]), err_center=r["point_center"] - t,
                 err_lin=r["point_lin"] - t, err_mid=r["point_mid"] - t,
                 sampling_rmse_equal_flops=float(np.sqrt(t * (1 - t) / n)))
    out.append(log)
    print(json.dumps(log[-1]), flush=True)
json.dump(out, open(f"results_noisy0025_{mf:.0e}.json", "w"), indent=1)
