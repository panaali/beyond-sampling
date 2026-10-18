"""Review of T241 claims B2 (n5/n9 delayed comparators), A4 (most damaging mode) and C2 (noise-injection precision).
Own code and seeds (Philox 21-23). usage: python checks.py -> results/checks.json"""
import json
import os

import numpy as np

import rv

out = {}
rng = np.random.default_rng(2024)

# ---- B2: n5 / n9. pre_t[j] = Wh[j] . h_{t-1} + Wi[j] x_{t-1} exactly; how much of it do the comparator columns carry?
x = rv.gauss(200_000, 21); pre, h = rv.states(x)
b2 = {"Wh_row": {f"n{j}": {f"n{k}": round(float(rv.Wh[j, k]), 3) for k in range(rv.H) if abs(rv.Wh[j, k]) > 0.05}
                 for j in (5, 9)}, "Wi": {"n5": float(rv.Wi[5]), "n9": float(rv.Wi[9])}}
for j in (5, 9):
    rows = {}
    for name, S in (("comparators n1,6,7,8 (exact weights, no fit)", rv.CMP),
                    ("comparators + n4 (exact weights)", rv.CMP + [4]),
                    ("all except comparators (exact weights)", [k for k in range(rv.H) if k not in rv.CMP])):
        r = []
        for t in range(2, rv.T + 1):
            part = h[:, t - 1][:, S] @ rv.Wh[j, S] + (rv.Wi[j] * x[:, t - 1] if "except" not in name else 0)
            if "except" in name:
                part = h[:, t - 1][:, S] @ rv.Wh[j, S]
            r.append(float(rv.r2(pre[:, t, j], part)))
        rows[name] = {"t2..10": [round(v, 5) for v in r], "t10": round(r[-1], 5)}
    # least-squares on lag-1 comparator states + x_t only (the "delayed comparator" regression on TRUE states)
    t = rv.T
    A = np.column_stack([h[:, t - 1][:, rv.CMP], x[:, t - 1]])
    coef = np.linalg.lstsq(A, pre[:, t, j], rcond=None)[0]
    rows["lstsq on true h_{t-1}[cmp] + x_t (t=10)"] = {"R2": float(rv.r2(pre[:, t, j], A @ coef)),
                                                       "coef": [round(float(c), 3) for c in coef]}
    b2[f"n{j}"] = rows
out["B2"] = b2
print("B2", json.dumps(b2)[:1500], flush=True)

# ---- A4: modes of the most common final gate pattern, resample of each mode's oblique coordinates + controls
x = rv.gauss(400_000, 22); pre, h = rv.states(x)
y = rv.answer(x); ref = (h[:, rv.T] @ rv.Wo.T).argmax(1)
pat = (pre[:, rv.T] > 0) @ (1 << np.arange(rv.H))
vals, cnt = np.unique(pat, return_counts=True)
a4 = []
for g in vals[np.argsort(-cnt)][:3]:
    idx = np.flatnonzero(pat == g); act = np.flatnonzero((int(g) >> np.arange(rv.H)) & 1)
    lam, R = np.linalg.eig(rv.Wh[np.ix_(act, act)]); L = np.linalg.inv(R)
    done, modes = set(), []
    for k in np.argsort(-np.abs(lam)):
        if k in done:
            continue
        ks = [k] if abs(lam[k].imag) < 1e-9 else [k, min((m for m in range(len(lam)) if m not in done and m != k),
                                                         key=lambda m: abs(lam[m] - np.conj(lam[k])))]
        done.update(ks); modes.append(ks)
    donor = rng.permutation(idx)
    H10, D10 = h[idx, rv.T][:, act], h[donor, rv.T][:, act]
    Hc = H10 - H10.mean(0)
    base_acc = float((ref[idx] == y[idx]).mean())

    def resample(P):
        hh = h[idx, rv.T].copy(); hh[:, act] = H10 + (D10 - H10) @ P.T
        p = (hh @ rv.Wo.T).argmax(1)
        return float((p == y[idx]).mean()), float((p == ref[idx]).mean())
    rec = {"pattern": int(g), "share": float(len(idx) / len(x)), "acc_in_pattern": base_acc,
           "cond_R": float(np.linalg.cond(R)), "modes": []}
    for ks in modes:
        P = (R[:, ks] @ L[ks]).real; l = lam[ks[0]]
        acc, agr = resample(P)
        rec["modes"].append({"abs": round(float(abs(l)), 3),
                             "period": round(float(2 * np.pi / abs(np.angle(l))), 2) if abs(l.imag) > 1e-9 else None,
                             "dim": len(ks), "energy": round(float(((Hc @ P.T) ** 2).sum() / (Hc ** 2).sum()), 4),
                             "acc": round(acc, 3), "agree": round(agr, 3)})
    # control: random orthogonal projectors of dim 1 and 2 in the active coordinates
    ctrl = {}
    for dim in (1, 2):
        accs, ens = [], []
        for _ in range(30):
            Q = np.linalg.qr(rng.standard_normal((len(act), dim)))[0]; P = Q @ Q.T
            accs.append(resample(P)[0]); ens.append(float(((Hc @ P.T) ** 2).sum() / (Hc ** 2).sum()))
        ctrl[f"random_dim{dim}"] = {"acc_mean": round(float(np.mean(accs)), 3), "acc_min": round(float(np.min(accs)), 3),
                                    "energy_mean": round(float(np.mean(ens)), 4)}
    # control 2: readout-aligned direction (the direction the readout is most sensitive to, dim 1)
    U = np.linalg.svd(rv.Wo[:, act])[2][:2].T
    ctrl["readout_top1"] = {"acc": round(resample(U[:, :1] @ U[:, :1].T)[0], 3)}
    ctrl["readout_top2"] = {"acc": round(resample(U @ U.T)[0], 3)}
    rec["controls"] = ctrl
    a4.append(rec)
    print("A4", g, rec["share"], base_acc, [(m["period"], m["abs"], m["energy"], m["acc"]) for m in rec["modes"]], ctrl,
          flush=True)
out["A4"] = a4

# ---- C2: noise needed for 99% agreement
x = rv.gauss(200_000, 23); pre, h = rv.states(x)
ref = (h[:, rv.T] @ rv.Wo.T).argmax(1)
sd = pre[:, rv.T].std(0)
c2 = {"final_step_only": []}
for eps in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6):
    p = np.maximum(pre[:, rv.T] + rng.standard_normal(pre[:, rv.T].shape) * sd * np.sqrt(eps), 0) @ rv.Wo.T
    c2["final_step_only"].append({"1-R2": eps, "agree": round(float((p.argmax(1) == ref).mean()), 4)})
    # (Noise at earlier steps is irrelevant for the all-16 replacement: with every neuron replaced, h_10 =
    # ReLU(f_10(x)) does not depend on earlier states, so this final-step test is the all-16 case.)
out["C2"] = c2
print("C2", c2, flush=True)
json.dump(out, open(os.path.join(rv.HERE, "results", "checks.json"), "w"), indent=1)
