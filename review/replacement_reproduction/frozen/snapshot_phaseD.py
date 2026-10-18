"""Phase D: where does the fully replaced model (all 16 neurons by formulas) disagree with the network?

Reads data/t241/phaseC2_eval.npz (eval inputs, network prediction ref, full-replacement prediction p_all).
Stratifies the disagreement by: network correct/wrong, case, blind region, the global 1st-2nd and 2nd-3rd gaps,
and how common the input's full gate pattern is (counted on the same eval set).
"""
import json
import numpy as np
from nacommon import run, target, cases

d = np.load("../../../data/t241/phaseC2_eval.npz")
x, ref, p = d["x"], d["ref"], d["p_all"]
y = target(x)
dis = p != ref
s = -np.sort(-x, 1)
g12, g23 = s[:, 0] - s[:, 1], s[:, 1] - s[:, 2]
C = cases(x)
out = {"N": len(x), "disagree": float(dis.mean())}
net_ok, rep_ok = ref == y, p == y
out["confusion"] = {"net_ok&rep_ok": float((net_ok & rep_ok).mean()), "net_ok&rep_wrong": float((net_ok & ~rep_ok).mean()),
                    "net_wrong&rep_ok": float((~net_ok & rep_ok).mean()),
                    "net_wrong&rep_wrong_same": float((~net_ok & ~rep_ok & ~dis).mean()),
                    "net_wrong&rep_wrong_diff": float((~net_ok & ~rep_ok & dis).mean())}
out["disagree_given"] = {"net_correct": float(dis[net_ok].mean()), "net_wrong": float(dis[~net_ok].mean())}
out["by_case"] = {k: {"share": float(m.mean()), "disagree": float(dis[m].mean()), "share_of_disagreements": float(dis[m].sum() / dis.sum())}
                  for k, m in C.items()}


def binned(v, edges, name):
    r = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi)
        r.append({"bin": [lo, hi], "share": float(m.mean()), "disagree": float(dis[m].mean()) if m.any() else None,
                  "net_err": float((~net_ok)[m].mean()) if m.any() else None,
                  "share_of_disagreements": float(dis[m].sum() / dis.sum())})
    out[name] = r


E = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, np.inf]
binned(g23, E, "by_gap23")
binned(g12, E, "by_gap12")
near = (g23 < 0.05) | (g12 < 0.05) | C["blind"]
out["near_tie_or_blind"] = {"share": float(near.mean()), "disagree": float(dis[near].mean()),
                            "disagree_elsewhere": float(dis[~near].mean()), "share_of_disagreements": float(dis[near].sum() / dis.sum())}
# gate-pattern frequency
_, pre = run(x)
gates = (pre[:, 1:] > 0).reshape(len(x), -1)
key = np.packbits(gates, axis=1)
_, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
inv = inv.ravel()
freq = cnt[inv]
binned(freq.astype(float), [1, 2, 5, 20, 100, 1000, np.inf], "by_full_gate_pattern_count")
for nm, msk in (("near", near), ("far", ~near)):
    out[f"by_full_gate_pattern_count_{nm}"] = [
        {"bin": [lo, hi], "share": float((msk & (freq >= lo) & (freq < hi)).mean()),
         "disagree": float(dis[msk & (freq >= lo) & (freq < hi)].mean()),
         "net_err": float((~net_ok)[msk & (freq >= lo) & (freq < hi)].mean())}
        for lo, hi in ((1, 2), (2, 5), (5, 20), (20, 100), (100, np.inf))]
out["far_disagree_given"] = {"net_correct": float(dis[~near & net_ok].mean()), "net_wrong": float(dis[~near & ~net_ok].mean()),
                             "net_err_far": float((~net_ok)[~near].mean())}
out["n_patterns"] = int(len(cnt))
# what does the replaced model pick when it disagrees?
r = np.argsort(np.argsort(-x, 1), 1)   # rank of each position
rk_p = r[np.arange(len(x)), p]
rk_ref = r[np.arange(len(x)), ref]
out["rank_picked_when_disagree"] = {"replacement": np.bincount(rk_p[dis], minlength=10).tolist(),
                                    "network": np.bincount(rk_ref[dis], minlength=10).tolist()}
json.dump(out, open("results/phaseD.json", "w"), indent=1)
print(json.dumps(out, indent=1))
