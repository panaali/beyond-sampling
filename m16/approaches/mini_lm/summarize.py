"""Baseline table (median and range over seeds) from models/*.json -> results_baselines.md."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "models", "*.json")))]
cols = [("large", "acc"), ("large", "illegal_mass"), ("large", "B0.5"), ("large", "B0.9"), ("large", "B0.99"),
        ("small_exact", "acc"), ("small_exact", "B0.5"), ("small_exact", "B0.9")]
lines = ["| lang | arch | params | FLOPs/FPE (large) | " + " | ".join(f"{h} {k}" for h, k in cols) + " | P7 max |z| |",
         "|" + "---|" * (5 + len(cols))]
for lang in ("dyck2", "modsum"):
    for arch in ("rnn", "softmax", "hard", "relu", "linear", "pos"):
        rs = [r for r in R if r["name"].startswith(f"{lang}_{arch}_s")]
        if not rs:
            continue
        cells = []
        for h, k in cols:
            v = [r[h][k] for r in rs if h in r]
            cells.append(f"{np.median(v):.3g} [{min(v):.3g}, {max(v):.3g}]" if v else "-")
        z = []
        for r in rs:  # P7: exact small horizon vs 1e6-sample estimate
            if "small_exact" in r:
                for k in ("B0.5", "B0.9"):
                    e, s = r["small_exact"][k], r["small_sampled_1e6"][k]
                    se = max(np.sqrt(max(e * (1 - e), 0) / 1e6), 1e-12)
                    z.append(abs(s - e) / se if (e > 0 or s > 0) else 0.0)
        lines.append(f"| {lang} | {arch} | {rs[0]['params']} | {rs[0]['flops_per_fpe_large']:.3g} | " + " | ".join(cells)
                     + f" | {max(z):.2f} |" if z else " | - |")
open(os.path.join(HERE, "results_baselines.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
