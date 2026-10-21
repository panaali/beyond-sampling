"""Bounded local pilot. This harness is NOT submission code.

NumPy is used only for references, diagnostics, serialization, and checks.
Every estimator call runs through the unmodified official meter.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, SetupContext

from estimator import Estimator

HERE = Path(__file__).resolve().parent
PIN = "96e308c518c76a1ae119f1d4c8913bc2e019eee4"


def limits():
    r = resource.getrusage(resource.RUSAGE_SELF)
    cpu = r.ru_utime + r.ru_stime
    rss_bytes = r.ru_maxrss if sys.platform == "darwin" else r.ru_maxrss * 1024
    if cpu > 580 or rss_bytes > 1_900_000_000:
        raise RuntimeError(f"Resource stop: CPU={cpu:.1f}s RSS={rss_bytes}")
    return {"cpu_seconds": cpu, "peak_rss_bytes": rss_bytes}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference(mlp, seed, count=65536):
    # Independent RNG stream and implementation. Marginal sample variances / N
    # estimate reference MSE, not uncertainty of the aggregate MSE statistic.
    rng = np.random.default_rng(seed)
    weights = [np.asarray(w) for w in mlp.weights]
    sums = np.zeros((mlp.depth, mlp.width), dtype=np.float64)
    final_second = np.zeros(mlp.width, dtype=np.float64)
    start = time.perf_counter()
    for offset in range(0, count, 2048):
        limits()
        x = rng.standard_normal((min(2048, count-offset), mlp.width), dtype=np.float32)
        for i, w in enumerate(weights):
            x = np.maximum(x @ w, 0)
            sums[i] += x.sum(axis=0, dtype=np.float64)
        final_second += np.square(x.astype(np.float64)).sum(axis=0)
    mean = sums / count
    var = (final_second - count * mean[-1]**2) / (count-1)
    return mean, var / count, time.perf_counter()-start


def evaluate(est, mlp, budget):
    est.setup(SetupContext(width=mlp.width, depth=mlp.depth, flop_budget=budget,
                          api_version="1.0", seed=8675309, submission_dir=str(HERE)))
    limits()
    start = time.perf_counter()
    with flops.BudgetContext(flop_budget=budget, wall_time_limit_s=120, quiet=True) as meter:
        pred = est.predict(mlp, budget)
    elapsed = time.perf_counter()-start
    est.teardown()
    pred = np.asarray(pred)
    assert pred.shape == (mlp.depth, mlp.width) and np.isfinite(pred).all()
    return pred, {"flops": meter.flops_used, "wall_seconds": elapsed,
                  "residual_seconds": meter.residual_wall_time_s,
                  "contract_pass": meter.residual_wall_time_s <= 0.4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starterkit", type=Path, default=Path("/tmp/whest-t473"))
    ap.add_argument("--stage", choices=["dev", "heldout", "full-dev", "full-heldout"], required=True)
    ap.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()
    assert subprocess.check_output(["git", "-C", str(args.starterkit), "rev-parse", "HEAD"], text=True).strip() == PIN
    versions = {p: importlib.metadata.version(p) for p in ("flopscope", "whestbench", "numpy")}
    assert versions["flopscope"] == "0.12.3" and versions["whestbench"] == "0.16.2"
    local = load_module(args.starterkit / "local_engine.py", "t473_local_engine")
    baselines = {name: load_module(args.starterkit / "examples" / file, "t473_"+name).Estimator
                 for name, file in [("mean", "02_mean_propagation.py"),
                                    ("covariance", "03_covariance_propagation.py")]}
    full = args.stage.startswith("full")
    heldout = args.stage.endswith("heldout")
    shapes = [(1024, 16)] if full else [(32, 4), (64, 16)]
    seeds = ([100] if heldout else [0]) if full else ([100, 101, 102, 103] if heldout else [0, 1])
    n, reps = (6500, 1) if full else (512, 4)
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (args.stage + ".json")
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}; retain every run")
    report = {"stage": args.stage, "starterkit": PIN, "versions": versions,
              "estimator_sha256": hashlib.sha256((HERE / "estimator.py").read_bytes()).hexdigest(),
              "reference_samples": 65536, "samples": n, "replicates": reps, "models": []}
    try:
        for width, depth in shapes:
            for seed in seeds:
                limits()
                budget = 2**41 if full else 10 * n * depth * 2 * width**2
                mlp = local.build_mlp(width, depth, seed)
                ref, ref_var, ref_time = reference(mlp, 1_000_000 + seed + width*1000)
                row = {"width": width, "depth": depth, "weight_seed": seed,
                       "budget": budget, "reference_wall_seconds": ref_time,
                       "reference_mse_estimate": float(ref_var.mean()), "estimates": []}
                arrays = {"reference": ref, "reference_variance": ref_var}
                jobs = [(name, cls(), 0) for name, cls in baselines.items()]
                jobs += [(name, Estimator(n, correction=correct), rep)
                         for rep in range(reps) for name, correct in [("mc", False), ("cv", True)]]
                for name, est, rep in jobs:
                    # Separate weight and input RNG seeds; repeat IDs don't
                    # depend on reference results. Same seed pairs MC and CV.
                    input_seed = 2_000_000 + seed*100 + rep
                    target = MLP(width=width, depth=depth, weights=mlp.weights, seed=input_seed)
                    pred, stats = evaluate(est, target, budget)
                    arrays[f"{name}_{rep}"] = pred
                    if not stats["contract_pass"]:
                        scored = np.zeros_like(pred)
                        multiplier = 1.0
                    else:
                        scored = pred
                        multiplier = max(0.1, stats["flops"] / budget)
                    mse = float(np.mean((scored[-1] - ref[-1])**2))
                    stats.update(method=name, replicate=rep, input_seed=input_seed,
                                 final_mse=mse, score=mse*multiplier,
                                 noise_corrected_score=(mse-ref_var.mean())*multiplier)
                    row["estimates"].append(stats)
                report["models"].append(row)
                np.savez_compressed(outdir / f"{args.stage}_w{width}_d{depth}_s{seed}.npz", **arrays)
                report["resources"] = limits()
                path.write_text(json.dumps(report, indent=2)+"\n")
                scores = {name: np.mean([e["score"] for e in row["estimates"] if e["method"] == name])
                          for name in ("mc", "cv", "mean", "covariance")}
                print(f"{args.stage} {width}x{depth} seed={seed} " +
                      " ".join(f"{k}={v:.3g}" for k,v in scores.items()), flush=True)
    except Exception as exc:
        report["stopped"] = repr(exc)
        path.write_text(json.dumps(report, indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
