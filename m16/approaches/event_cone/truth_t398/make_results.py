"""Pool runs/t<t0>_*.jsonl into results.json (est, se, n_drawn, n_lines, sec per t0) and check the t0 = 1.5 gate."""
import json
import os

import numpy as np

import truth

REF = 2.16e-7; REF_SE = float(np.sqrt(REF / 1e9))            # T340 CPU plain MC, 1e9 inputs
GPU = 2.09e-7; GPU_SE = 4.571651780264984e-09                # T340 GPU plain MC, 1e10 inputs

out = {}
for t0 in (1.5, 2.0, 2.25, 2.5):
    s = truth.summary(truth.load(t0)); s["rel_se"] = s["se"] / s["est"] if s["est"] > 0 else None
    out[str(t0)] = s
v = out["1.5"]
v["validation"] = dict(ref=REF, ref_se=REF_SE, z_diff=(v["est"] - REF) / np.hypot(v["se"], REF_SE),
                       z_strict=(v["est"] - REF) / v["se"], z_gpu=(v["est"] - GPU) / np.hypot(v["se"], GPU_SE))
v["validation"]["pass"] = bool(abs(v["validation"]["z_diff"]) <= 2)
out["cpu_sec_t0_2.0_to_2.5"] = sum(out[k]["sec"] for k in ("2.0", "2.25", "2.5"))
json.dump(out, open(os.path.join(truth.HERE, "results.json"), "w"), indent=1)
for k, s in out.items():
    if isinstance(s, dict):
        print(k, f"est {s['est']:.4g} se {s['se']:.3g} rel {s['rel_se']:.3f} nz {s['n_nonzero']} lines {s['n_lines']} "
                 f"drawn {s['n_drawn']} sec {s['sec']:.0f}")
print("validation", out["1.5"]["validation"], "CPU s (2.0-2.5)", round(out["cpu_sec_t0_2.0_to_2.5"]))
