"""DRI-owned leaderboard: every estimator evaluated on identical models, truth and compute accounting.

Approach folders that want to be ranked expose `m16/approaches/<name>/estimator.py` with
    est(model, budget, rng) -> (estimate, cost)
where model = (Wi, Wh, Wo), budget and cost are in *forward-pass equivalents*: one forward pass of the RNN
on one length-10 sequence = 1. Count ALL work that depends on the model (model passes, gate/slope
passes, LPs scaled by FLOPs/6e3), and input generation/rejection if not negligible. Model-free
precomputation that is reused across models is free but must be declared in the README.
Set `DEDUCTIVE = True` in estimator.py only if the estimate uses no random model inputs.

usage: python m16/leaderboard.py truths            # cache ground truth (16M samples per model)
       python m16/leaderboard.py run name1 name2   # evaluate approaches (or 'sampling', 'stratified')
"""
import importlib.util
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, "m16")
from arc_eval import Wi0, Wh0, Wo0, est_sampling, forward, noisy_models  # noqa: E402

TRUTHS = "results/leaderboard_truths.json"
BUDGETS = (1_000, 4_000, 16_000, 64_000, 256_000)
REPS = 8


def models():
    ms = {"orig": (Wi0, Wh0, Wo0)}
    for sigma in (0.0025, 0.005):
        for k, m in enumerate(noisy_models(6, sigma, seed=0)):
            ms[f"s{sigma}_{k}"] = m
    return ms


def compute_truths(n=16_000_000, chunk=1_000_000):
    out = {}
    for name, m in models().items():
        rng = np.random.default_rng(987_654); ok = 0
        for _ in range(n // chunk):
            x = rng.standard_normal((chunk, 10))
            ok += int((forward(m, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
        p = ok / n
        out[name] = {"acc": p, "se": float(np.sqrt(p * (1 - p) / n)), "n": n}
        print(name, out[name], flush=True)
    json.dump(out, open(TRUTHS, "w"), indent=1)


def load_estimator(name):
    if name == "sampling":
        return est_sampling, False
    if name == "stratified":
        import stratified as S
        d = json.load(open("m16/stratified_design.json"))
        P, alloc = np.array(d["P"]), np.array(d["alloc"])
        return (lambda m, b, rng: S.estimate(lambda x: forward(m, x), b, rng, P, alloc)), False
    path = f"m16/approaches/{name}/estimator.py"
    spec = importlib.util.spec_from_file_location(f"est_{name}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.est, bool(getattr(mod, "DEDUCTIVE", False))


def run(names):
    truth = json.load(open(TRUTHS)); ms = models(); rows = {}
    for name in names:
        est, deductive = load_estimator(name)
        for b in BUDGETS:
            for group in ("orig", "s0.0025", "s0.005"):
                se, cost = [], []
                for mname, m in ms.items():
                    if not mname.startswith(group):
                        continue
                    for r in range(1 if deductive else REPS):
                        e, c = est(m, b, np.random.default_rng(10_000 * r + zlib.crc32(mname.encode()) % 1000))
                        se.append((e - truth[mname]["acc"]) ** 2); cost.append(c)
                p = np.mean([truth[k]["acc"] for k in ms if k.startswith(group)])
                C = float(np.mean(cost)); mse = float(np.mean(se))
                ratio = mse / (p * (1 - p) / C)      # < 1 means better than plain sampling at equal compute
                rows[f"{name}|{group}|{b}"] = dict(cost=C, rmse_pp=100 * np.sqrt(mse), mse_ratio_vs_sampling=ratio,
                                                   deductive=deductive)
                print(f"{name:18s} {group:8s} budget {b:7d} cost {C:9.0f} RMSE {100*np.sqrt(mse):.3f}pp "
                      f"MSE/sampling {ratio:.2f}", flush=True)
            if deductive:
                break
    old = json.load(open("results/leaderboard.json")) if os.path.exists("results/leaderboard.json") else {}
    old.update(rows); json.dump(old, open("results/leaderboard.json", "w"), indent=1)


if __name__ == "__main__":
    {"truths": lambda: compute_truths(), "run": lambda: run(sys.argv[2:])}[sys.argv[1]]()
