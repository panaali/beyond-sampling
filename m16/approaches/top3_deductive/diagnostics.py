"""T408 design diagnostics for the top-k-conditioning route on D1w128 (2^30 seed 0); sampled, for design only.

(1) Where errors go: winners outside the top 3, and gate accuracy of the comparators that a top-3 case fixes.
(2) Size of the units a top-3 case leaves undetermined (both-bottom comparators, both-low units), in margin units.
(3) For one case prefix, condition on the ordering of the top k inputs (k = 3..10) and fit the best linear function of
    x to the margins z_2nd - z_1st and z_2nd - z_3rd (least squares on samples: more than a deductive rule could use).
    Report the residual and the error rate that the linear surrogate implies vs the true conditional error rate.

usage: python diagnostics.py   -> results/diagnostics.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../ffn_control"))
import analyze  # noqa: E402

T = 10


def main():
    m = analyze.load(os.path.join(HERE, "../ffn_control/models/D1w128_s0.json"))
    W1, W2 = m.W
    out = {}
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400_000, T))   # 400k keeps the (n, 128) arrays under the 2 GB cap
    pre = x @ W1.T
    h = np.maximum(pre, 0)
    z = h @ W2.T
    pred = z.argmax(1)
    o = np.argsort(-x, 1)
    r = np.arange(len(x))
    t1, t2, t3 = o[:, 0], o[:, 1], o[:, 2]
    out["error"] = float((pred != t2).mean())
    out["pred_top1"] = float((pred == t1).mean())
    out["pred_top3"] = float((pred == t3).mean())
    out["pred_outside_top3"] = float(((pred != t1) & (pred != t2) & (pred != t3)).mean())
    U = W1 / np.linalg.norm(W1, axis=1)[:, None]
    comp = (np.sort(np.abs(U), 1)[:, -2] > .6) & (U.max(1) > .6) & (U.min(1) < -.6)
    A, B = np.argmax(U, 1), np.argmin(U, 1)
    intop = np.zeros((len(x), T), bool)
    intop[r, t1] = intop[r, t2] = intop[r, t3] = True
    fixed = comp[None, :] & (intop[:, A] | intop[:, B])
    low = np.argsort(U, 1)[:, :2]
    bl_top = (~comp)[None, :] & (intop[:, low[:, 0]] | intop[:, low[:, 1]])
    g = pre > 0
    out["fixed_comparators_per_case"] = float(fixed.sum(1).mean())
    out["fixed_comparator_gate_error"] = float((g != (x[:, A] > x[:, B]))[fixed].mean())
    out["both_low_with_top_input_per_case"] = float(bl_top.sum(1).mean())
    out["both_low_with_top_input_on_rate"] = float(g[bl_top].mean())
    und = ~fixed & ~bl_top
    out["undetermined_units_per_case"] = float(und.sum(1).mean())
    zu = np.where(und, h, 0) @ W2.T
    out["margin_abs_median"] = [float(np.median(abs(z[r, t2] - z[r, t1]))), float(np.median(abs(z[r, t2] - z[r, t3])))]
    out["undetermined_contribution_sd"] = [float((zu[r, t2] - zu[r, t1]).std()), float((zu[r, t2] - zu[r, t3]).std())]
    out["topk"] = []
    rng = np.random.default_rng(4)
    n = 300_000
    for k in (3, 4, 5, 6, 7, 8, 10):
        s = -np.sort(-rng.standard_normal((n, T)), 1)
        xk = np.empty_like(s)
        xk[:, :k] = s[:, :k]                                   # case prefix: position p holds rank p for p < k
        perm = np.argsort(rng.random((n, T - k)), 1)
        for c in range(T - k):
            xk[:, k + c] = s[np.arange(n), k + perm[:, c]]
        zk = np.maximum(xk @ W1.T, 0) @ W2.T
        m1, m3 = zk[:, 1] - zk[:, 0], zk[:, 1] - zk[:, 2]
        X = np.c_[xk, np.ones(n)]
        res = [mm - X @ np.linalg.lstsq(X, mm, rcond=None)[0] for mm in (m1, m3)]
        err = ~((m1 > 0) & (m3 > 0) & (zk.argmax(1) == 1))
        err_lin = ~(((m1 - res[0]) > 0) & ((m3 - res[1]) > 0))
        out["topk"].append({"k": k, "cases": int(np.prod(range(T - k + 1, T + 1))),
                            "residual_sd": [float(res[0].std()), float(res[1].std())],
                            "P_err_given_case": float(err.mean()), "P_err_linear_surrogate": float(err_lin.mean())})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results/diagnostics.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
