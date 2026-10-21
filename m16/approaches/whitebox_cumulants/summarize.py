"""Score frozen predictions against pinned public means, no model runs."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REFS = Path("data/t474-references")
archived = RESULTS / "references"
archived.mkdir(exist_ok=True)
for path in REFS.glob("*"):
    if path.suffix in (".json", ".npz"):
        shutil.copyfile(path, archived / path.name)

records = []
for label in ("mini_0", "mini_1", "full_0", "full_1"):
    row = json.loads((archived / f"{label}.json").read_text())
    ref_path = archived / f"{label}.npz"
    ref = np.load(ref_path)["final_means"].astype(np.float64)
    record = {"row": label, "mlp_seed": row["mlp_seed"],
              "reference_archive_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
              "estimated_reference_mse": row["avg_variance"] / 1e9, "methods": {}}
    for method in ("blocked", "covariance"):
        meta = json.loads((RESULTS / f"{method}_{label}.json").read_text())
        assert meta["mlp_seed"] == row["mlp_seed"]
        predictions = np.load(RESULTS / f"{method}_{label}.npy")
        difference = predictions[-1].astype(np.float64) - ref
        raw_mse = float(np.mean(difference*difference))
        passed = meta["local_contract_pass"]
        score = raw_mse * max(.1, meta["flops"] / 2**41) if passed else float(np.mean(ref*ref))
        record["methods"][method] = {
            "raw_mse": raw_mse, "official_formula_score": score,
            "flops": meta["flops"], "cost_fraction": meta["cost_fraction"],
            "residual_seconds": meta["residual_seconds"], "local_contract_pass": passed,
            "peak_rss_bytes": max(meta["peak_rss_bytes"], meta["supervisor_peak_rss_bytes"]),
            "weights_sha256": meta["weights_sha256"],
        }
    assert record["methods"]["blocked"]["weights_sha256"] == record["methods"]["covariance"]["weights_sha256"]
    ratio = record["methods"]["blocked"]["official_formula_score"] / record["methods"]["covariance"]["official_formula_score"]
    record.update(score_ratio_to_covariance=ratio, strict_point_pass=ratio<=1, lenient_point_pass=ratio<=10)
    records.append(record)

parity = []
for label in ("32x4", "256x16", "512x16", "512x16_s2027"):
    a = np.load(RESULTS / f"v25_{label}.npy").astype(float)
    b = np.load(RESULTS / f"blocked_{label}.npy").astype(float)
    maximum = float(np.max(np.abs(a-b)))
    mse = float(np.mean((a[-1]-b[-1])**2))
    assert maximum <= 3e-5 and mse <= 1e-11
    parity.append({"case": label, "max_abs_difference": maximum, "last_layer_mse_difference": mse})

groups = {}
for split in ("mini", "full", "all"):
    selected = [r for r in records if split == "all" or r["row"].startswith(split)]
    scores = {m: float(np.mean([r["methods"][m]["official_formula_score"] for r in selected])) for m in ("blocked", "covariance")}
    ratio = scores["blocked"] / scores["covariance"]
    groups[split] = dict(models=len(selected), scores=scores, ratio=ratio,
                         strict_point_pass=ratio<=1, lenient_point_pass=ratio<=10,
                         population_ci=None)
    print(split, groups[split])

cpu = 0
for path in RESULTS.glob("*.json"):
    d = json.loads(path.read_text())
    if isinstance(d, dict):
        cpu += max(d.get("worker_cpu_seconds", 0), d.get("supervisor_cpu_seconds", 0))
report = {"dataset_revision": "aa99830fdc09fad15407b10e8e3459d3e18bba0a",
          "reference_samples_per_model": 1_000_000_000,
          "records": records, "groups": groups, "parity": parity,
          "worker_cpu_seconds_including_diagnostic": cpu,
          "peak_full_width_rss_bytes": max(r["methods"]["blocked"]["peak_rss_bytes"] for r in records),
          "scope": "Four public rows only, two mini DEV and two full held out from our development; not an official submission or leaderboard-equivalent score."}
(RESULTS / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
print("CPU",cpu,"full peak RSS",report["peak_full_width_rss_bytes"])
