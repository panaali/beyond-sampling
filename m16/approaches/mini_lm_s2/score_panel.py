"""T445 scoring: truth (4e6 MC) + R = 5 runs of each estimator at B = 1e3 and 1e4 per model.
Usage: python score_panel.py <arch> [seeds 100-109]  -> results/<model>.json (skips finished models)."""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mini_lm"))
import estimators as E  # noqa: E402
from load import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MLP1 = os.path.join(HERE, "..", "mini_lm", "models_mlp1")
N_TRUTH, CH, R, BUDGETS = 4_000_000, 100_000, 5, (1e3, 1e4)


def main():
    torch.set_num_threads(2)
    arch = sys.argv[1]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for seed in range(100, 110):
        name = f"dyck2_{arch}_s{seed}"
        out = os.path.join(HERE, "results", name + ".json")
        if os.path.exists(out):
            continue
        model, lang = load(name, MLP1)
        t0 = time.time()
        rng = np.random.default_rng(9_000_000 + seed)
        s1 = s2 = 0.0
        for _ in range(N_TRUTH // CH):
            Y = E.p_illegal(model, lang, lang.sample(CH, rng)).mean(1)
            s1 += Y.sum(); s2 += (Y ** 2).sum()
        Q = s1 / N_TRUTH
        res = dict(name=name, truth=Q, sigma2=s2 / N_TRUTH - Q ** 2, n_truth=N_TRUTH, truth_seconds=time.time() - t0, runs={})
        for k, f in E.ESTIMATORS.items():
            for B in BUDGETS:
                rr = []
                for r in range(R):
                    t1 = time.time()
                    est, c = f(model, lang, B, np.random.default_rng(10_000 * seed + 100 * r + int(np.log10(B))))
                    rr.append(dict(est=est, cost=c, seconds=time.time() - t1))
                res["runs"][f"{k}@{int(B)}"] = rr
        json.dump(res, open(out, "w"), indent=1)
        print(name, Q, res["sigma2"], round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
