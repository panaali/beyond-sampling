"""Phase A (modes): W_hh eigen-modes of the dominant final gate patterns, and mode-resample ablation.

Gate pattern g = active set of pre_10 (the final step). For the top-K patterns (by frequency on the eval set),
eigendecompose A_g = D_g W_hh restricted to the active neurons. Each real eigenvalue / complex pair defines a
1- or 2-dim invariant subspace with oblique coordinates from the left eigenvectors (sum of the projections = identity).
Ablation: replace the mode's coordinates of h_10 (or of h_9, then run step 10 exactly) by those of a random donor
input with the same final gate pattern; measure accuracy/agreement within the pattern.
"""
import json
import numpy as np
from nacommon import sample, run, run_from, target, Wi, Wh, Wo, T, H

N = 400_000
x = sample(N, 241011)
hs, pre = run(x)
ref = (hs[:, -1] @ Wo.T).argmax(1)
y = target(x)
gate = (pre[:, T] > 0) @ (1 << np.arange(H))
vals, counts = np.unique(gate, return_counts=True)
order = np.argsort(-counts)
K = 5
rng = np.random.default_rng(241012)
out = {"N": N, "patterns": []}
for gi in order[:K]:
    g = int(vals[gi])
    idx = np.where(gate == g)[0]
    act = np.array([j for j in range(H) if g >> j & 1])
    A = Wh[np.ix_(act, act)]
    lam, R = np.linalg.eig(A)
    L = np.linalg.inv(R)                    # rows = left eigenvectors, L R = I
    # group conjugate pairs
    used, modes = set(), []
    for k in np.argsort(-np.abs(lam)):
        if k in used:
            continue
        if abs(lam[k].imag) > 1e-9:
            kc = [m for m in range(len(lam)) if m not in used and m != k and abs(lam[m] - lam[k].conj()) < 1e-8][0]
            ks = [k, kc]
        else:
            ks = [k]
        used.update(ks)
        modes.append(ks)
    donor = rng.permutation(idx)
    rec = {"pattern": g, "share": float(len(idx) / N), "active": act.tolist(),
           "acc_in_pattern": float((ref[idx] == y[idx]).mean()), "modes": []}
    for ks in modes:
        P = (R[:, ks] @ L[ks, :]).real          # oblique projector onto the mode (active coords)
        l = lam[ks[0]]
        period = float(2 * np.pi / abs(np.angle(l))) if abs(l.imag) > 1e-9 else None
        load = np.zeros(H)
        load[act] = np.linalg.norm(R[:, ks], axis=1) / np.linalg.norm(R[:, ks])
        m = {"abs": float(abs(l)), "angle_deg": float(np.degrees(abs(np.angle(l)))), "period": period,
             "top_neurons": [f"n{j}({load[j]:.2f})" for j in np.argsort(-load)[:4]]}
        for step in (T, T - 1):
            h = hs[idx, step].copy()
            hd = hs[donor, step]
            h[:, act] = h[:, act] + (hd[:, act] - h[:, act]) @ P.T
            lg = h @ Wo.T if step == T else run_from(h, x[idx], step)
            p = lg.argmax(1)
            m[f"resample_h{step}"] = {"acc": float((p == y[idx]).mean()), "agree": float((p == ref[idx]).mean())}
            # energy of the mode coordinate (fraction of centered final-state variance removed by the projector)
        hc = hs[idx, T][:, act] - hs[idx, T][:, act].mean(0)
        m["energy_share"] = float(((hc @ P.T) ** 2).sum() / (hc ** 2).sum())
        rec["modes"].append(m)
    out["patterns"].append(rec)
    print(g, rec["share"], rec["acc_in_pattern"], flush=True)
    for m in rec["modes"]:
        print("  |l|=%.3f ang=%.1f per=%s E=%.3f h10 acc %.3f agr %.3f | h9 acc %.3f  %s" % (
            m["abs"], m["angle_deg"], m["period"] and round(m["period"], 2), m["energy_share"],
            m["resample_h10"]["acc"], m["resample_h10"]["agree"], m["resample_h9"]["acc"], m["top_neurons"]))
json.dump(out, open("results/phaseA_modes.json", "w"), indent=1)
