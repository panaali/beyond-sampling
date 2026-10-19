"""L = 32 scoring on MLP1 (T447): certified UB (depth-first branch and bound, frozen recipe) vs plain sampling's
two-sided 95% Clopper-Pearson upper bound at the same FPE. Usage: python score32.py models_dir name [name ...]
-> data/score32_<name>.json"""
import os, sys, json, time
import numpy as np
import torch
from scipy.stats import beta
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mini_lm"))
from s4load import load_any as load
from metrics import sampled
import cert, bab

torch.set_num_threads(1)
T = 32
AMAX = 1                       # frozen from dev: IBP closes exactly 1 position ahead (horizon.py)
PLAN = {0.9: 1e6, 0.5: 1e5, 0.99: 1e5}     # tau -> budget of the branch and bound run (snapshots at 1e5 and 1e6)
if os.environ.get("S4_PLAN"):
    PLAN = {float(a): float(b) for a, b in (x.split(":") for x in os.environ["S4_PLAN"].split(","))}


def cp_upper(x, n):
    return 1.0 if x >= n else float(beta.ppf(0.975, x + 1, n - x))


def main():
    mdir = sys.argv[1]
    for name in sys.argv[2:]:
        out_path = os.path.join(HERE, "data", f"score32_{name}" + os.environ.get("S4_TAG", "") + ".json")
        if os.path.exists(out_path):
            continue
        model, lang = load(name, models_dir=None if mdir == "-" else mdir)
        P = cert.Params(model)
        seed = int(name.rsplit("_s", 1)[1])
        out = dict(name=name, T=T, amax=AMAX, fpe_flops=model.flops(T), bab={}, sampling={})
        for N in (100000, 1000000):
            t0 = time.time()
            r = sampled(model, lang, N, seed=1000 + seed + (0 if N == 100000 else 500000))
            out["sampling"][str(N)] = {str(t): dict(events=int(round(r[f"B{t}"] * N)), B=r[f"B{t}"],
                                                    cp95_upper=cp_upper(int(round(r[f"B{t}"] * N)), N))
                                       for t in (0.5, 0.9, 0.99)}
            out["sampling"][str(N)]["sec"] = time.time() - t0
        for tau, budget in PLAN.items():
            r = bab.run_dfs(P, T=T, tau=tau, budget_fpe=budget, fpe_flops=model.flops(T), amax=AMAX, chunk=512,
                            snapshots=(1e5, 1e6))
            out["bab"][str(tau)] = r
        for tau in (0.5, 0.9, 0.99):
            for N in (100000, 1000000):
                snaps = {s["budget"]: s["ub"] for s in out["bab"][str(tau)]["snapshots"]}
                ub = snaps.get(float(N), out["bab"][str(tau)]["ub"] if out["bab"][str(tau)]["fpe"] >= N else None)
                cp = out["sampling"][str(N)][str(tau)]["cp95_upper"]
                out.setdefault("compare", {})[f"{tau}@{N}"] = dict(ub=ub, cp95=cp, strict=(ub is not None and ub <= cp),
                                                                   lenient=(ub is not None and ub <= 10 * cp))
        json.dump(out, open(out_path, "w"), indent=1)
        print(name, json.dumps(out["compare"]), flush=True)


if __name__ == "__main__":
    main()
