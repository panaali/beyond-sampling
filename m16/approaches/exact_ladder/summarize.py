"""Per-T summary table from est_info.jsonl (estimator log) + leaderboard truths. usage: python summarize.py"""
import hashlib
import json
import sys

import numpy as np

sys.path.append("../..")
import leaderboard as L  # noqa: E402

info = {}
for l in open("est_info.jsonl"):
    d = json.loads(l); info[(d["T"], d["key"])] = d
rows = []
for T in (2, 3, 4, 5):
    try:
        tr = json.load(open("../../../" + L.truths_path(16, T)))
    except FileNotFoundError:
        continue
    for name, m in L.models(16, T).items():
        key = hashlib.sha1(b"".join(np.ascontiguousarray(W).tobytes() for W in m)).hexdigest()[:12]
        d = info.get((T, key))
        if d is None:
            continue
        t = tr[name]; p = d["acc"]
        # MSE ratio bound: (numerical error bound)^2 / (p(1-p)/cost)
        rows.append(dict(T=T, name=name, group=L.group_of(name), acc=p, truth=t["acc"], z=(p - t["acc"]) / t["se"],
                         cells=d["cells"], wrong=d["wrong_pieces"], correct=d.get("correct_pieces"),
                         simp=d["simplices"], fpe=d["fpe"], err=d["err_est"], cpu=d["cpu_s"],
                         ratio_bound=d["err_est"] ** 2 / (p * (1 - p) / d["fpe"])))
json.dump(rows, open("summary.json", "w"), indent=1)
print(f"{'T':>2} {'group':8s} {'n':>3} {'cells':>9} {'wrong pcs':>9} {'simplices':>10} {'cost FPE':>10} {'err bound':>9} "
      f"{'ratio<=':>8} {'z mean':>6} {'z rms':>5} {'cpu s':>6}")
for T in sorted({r["T"] for r in rows}):
    for g in ["seed", "s0.0025", "s0.005", "s0.01", "s0.02", "all"]:
        s = [r for r in rows if r["T"] == T and (g == "all" or r["group"] == g)]
        if not s:
            continue
        z = np.array([r["z"] for r in s])
        print(f"{T:2d} {g:8s} {len(s):3d} {np.mean([r['cells'] for r in s]):9.0f} {np.mean([r['wrong'] for r in s]):9.0f} "
              f"{np.mean([r['simp'] for r in s]):10.0f} {np.mean([r['fpe'] for r in s]):10.3g} "
              f"{max(r['err'] for r in s):9.2g} {max(r['ratio_bound'] for r in s):8.1e} {z.mean():+6.2f} "
              f"{np.sqrt((z**2).mean()):5.2f} {np.mean([r['cpu'] for r in s]):6.0f}")
