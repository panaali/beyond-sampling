"""Aggregate variance_<sigma>.json (from variance_calc.py) into copy-averaged gains at the final 1.7/sample charge."""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); f = 1.6 / 1.7
for lv in ("0", "0.0025", "0.005"):
    d = json.load(open(os.path.join(HERE, f"variance_{lv}.json")))
    vs = np.array([r["acc"] * (1 - r["acc"]) for r in d])
    agg = lambda k: vs.mean() / (vs / np.array([r[k] for r in d])).mean()
    print(lv, "strat %.2f" % agg("gain_strat"), "rb %.2f" % agg("gain_rb"), "srb %.2f" % (agg("gain_srb") * f),
          "srb/strat %.2f" % (agg("gain_srb") * f / agg("gain_strat")),
          "per-copy srb/strat %.2f-%.2f" % (min(r["srb_vs_strat"] for r in d) * f, max(r["srb_vs_strat"] for r in d) * f),
          "per-copy srb %.2f-%.2f" % (min(r["gain_srb"] for r in d) * f, max(r["gain_srb"] for r in d) * f))
