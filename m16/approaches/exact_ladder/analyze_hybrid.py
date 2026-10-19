"""Tables for the T46 arm-B scoring from score_T{T}.jsonl. usage: python analyze_hybrid.py T
MSE per model = mean over rule seeds of (est - truth)^2 - truth_se^2; ratio = group mean MSE / group mean of
p(1-p)/cost (leaderboard v3 formula). T=5 truth: exact (est_info.jsonl) when available, else the leaderboard 4M truth."""
import hashlib
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.append("../..")
import leaderboard as L  # noqa: E402

T = int(sys.argv[1])
tr = json.load(open("../../../" + L.truths_path(16, T)) if T != 6 else open("truths_H16_T6_from_t10.json"))
exact = {}
for l in open("est_info.jsonl"):
    d = json.loads(l)
    if d["T"] == T:
        exact[d["key"]] = d["acc"]
ms = L.models(16, T)
truth = {}
for n, m in ms.items():
    k = hashlib.sha1(b"".join(np.ascontiguousarray(W).tobytes() for W in m)).hexdigest()[:12]
    truth[n] = (exact[k], 0.0, "exact") if k in exact else (tr[n]["acc"], tr[n]["se"], "4M")
rows = [json.loads(l) for l in open(f"score_T{T}.jsonl")]
by = defaultdict(list)
for r in rows:
    by[(r["name"], r["B"])].append(r)
res = {}
print(f"T={T}  truths: {sum(v[2] == 'exact' for v in truth.values())} exact, {sum(v[2] == '4M' for v in truth.values())} 4M")
print(f"{'B':>6} {'group':8} {'n':>3} {'cost':>8} {'exactFPE':>8} {'cells':>5} {'wrongE/wrong':>12} "
      f"{'hyb/samp':>8} {'walk/samp':>9} {'hyb/walk':>8} {'rmse hyb':>9} {'rmse walk':>9} {'perseed hyb/samp':>30}")
for B in (16_000, 64_000):
    for g in ("seed", "s0.0025", "s0.005", "s0.01", "s0.02"):
        names = [n for n in ms if L.group_of(n) == g and (n, B) in by]
        if not names:
            continue
        mh, mw, ref, perseed = [], [], [], []
        for n in names:
            t, se, _ = truth[n]; rr = by[(n, B)]
            eh = np.mean([(r["hybrid"] - t) ** 2 for r in rr]) - se ** 2
            ew = np.mean([(r["walk"] - t) ** 2 for r in rr]) - se ** 2
            rf = np.mean([t * (1 - t) / r["hybrid_cost"] for r in rr])
            mh.append(eh); mw.append(ew); ref.append(rf); perseed.append(eh / rf)
        r0 = by[(names[0], B)]
        cost = np.mean([np.mean([r["hybrid_cost"] for r in by[(n, B)]]) for n in names])
        efpe = np.mean([by[(n, B)][0]["exact_fpe"] for n in names])
        cells = np.mean([by[(n, B)][0]["cells"] for n in names])
        share = np.mean([by[(n, B)][0]["exact_wrong"] / (1 - truth[n][0]) for n in names])
        out = dict(n=len(names), cost=cost, exact_fpe=efpe, cells=cells, wrong_share=share,
                   hyb_ratio=np.mean(mh) / np.mean(ref), walk_ratio=np.mean(mw) / np.mean(ref),
                   hyb_over_walk=np.mean(mh) / np.mean(mw), rmse_h=np.sqrt(max(np.mean(mh), 0)),
                   rmse_w=np.sqrt(max(np.mean(mw), 0)), per_model=[round(x, 3) for x in perseed])
        res[f"{B}_{g}"] = out
        ps = " ".join(f"{x:.2f}" for x in perseed) if g == "seed" else ""
        print(f"{B:6d} {g:8} {len(names):3d} {cost:8.3g} {efpe:8.3g} {cells:5.1f} {share:12.3f} {out['hyb_ratio']:8.3f} "
              f"{out['walk_ratio']:9.3f} {out['hyb_over_walk']:8.3f} {out['rmse_h']:9.2e} {out['rmse_w']:9.2e} {ps:>30}")
json.dump(res, open(f"hybrid_T{T}.json", "w"), indent=1)
