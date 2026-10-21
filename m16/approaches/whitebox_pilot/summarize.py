"""Rescore saved predictions independently of pilot.py; no model evaluations."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
ROOT = parser.parse_args().results_dir
groups = {}
total_cpu = 0
max_rss = 0
for filename in ("dev", "heldout", "full-dev", "full-heldout"):
    doc = json.loads((ROOT / f"{filename}.json").read_text())
    total_cpu += doc["resources"]["cpu_seconds"]
    max_rss = max(max_rss, doc["resources"]["peak_rss_bytes"])
    for model in doc["models"]:
        w, d, seed = model["width"], model["depth"], model["weight_seed"]
        saved = np.load(ROOT / f"{filename}_w{w}_d{d}_s{seed}.npz")
        methods = {}
        for method in ("mc", "cv", "mean", "covariance"):
            entries = [e for e in model["estimates"] if e["method"] == method]
            scores, corrected, estimates = [], [], []
            for e in entries:
                pred = saved[f"{method}_{e['replicate']}"]
                factor = max(0.1, e["flops"] / model["budget"])
                if not e["contract_pass"]:
                    pred, factor = np.zeros_like(pred), 1.0
                mse = np.mean((pred[-1] - saved["reference"][-1])**2)
                np.testing.assert_allclose(mse * factor, e["score"], rtol=1e-12)
                scores.append(float(mse * factor))
                corrected.append(float((mse-saved["reference_variance"].mean())*factor))
                estimates.append(pred[-1])
            methods[method] = {
                "score": float(np.mean(scores)),
                "noise_corrected_score": float(np.mean(corrected)),
                "flops": entries[0]["flops"],
                "replicate_variance": float(np.var(estimates, axis=0, ddof=1).mean()) if len(estimates)>1 else None,
                "all_contract_pass": all(e["contract_pass"] for e in entries),
            }
        key = f"{filename}:{w}x{d}"
        groups.setdefault(key, []).append(methods)

summary = {"groups": {}, "cpu_seconds_total": total_cpu, "peak_rss_bytes": max_rss}
for key, rows in groups.items():
    scores = {method: float(np.mean([r[method]["score"] for r in rows]))
              for method in rows[0]}
    cv_mc = scores["cv"] / scores["mc"]
    summary["groups"][key] = {
        "models": len(rows), "scores": scores,
        "cv_to_mc": cv_mc, "cv_to_covariance": scores["cv"]/scores["covariance"],
        "strict_point_pass": cv_mc <= 1, "lenient_point_pass": cv_mc <= 10,
        "noise_corrected_scores": {m: float(np.mean([r[m]["noise_corrected_score"] for r in rows])) for m in rows[0]},
        "flops_per_model": {m: rows[0][m]["flops"] for m in rows[0]},
        "all_contract_pass": all(v["all_contract_pass"] for r in rows for v in r.values()),
    }
    # Four held-out networks are too few for a strong population conclusion.
    # Model-cluster bootstrap is descriptive; never resample neurons as IID.
    if len(rows) >= 4:
        rng = np.random.default_rng(473)
        picks = rng.integers(0, len(rows), size=(10000, len(rows)))
        cv = np.array([r["cv"]["score"] for r in rows])
        mc = np.array([r["mc"]["score"] for r in rows])
        ratios = cv[picks].mean(axis=1) / mc[picks].mean(axis=1)
        summary["groups"][key]["descriptive_model_bootstrap_cv_mc_95"] = np.quantile(ratios, [0.025, 0.975]).tolist()
    print(key, " ".join(f"{m}={s:.6g}" for m,s in scores.items()),
          f"CV/MC={cv_mc:.4f} CV/cov={scores['cv']/scores['covariance']:.4f}")
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print("CPU seconds", total_cpu, "peak RSS bytes", max_rss)
