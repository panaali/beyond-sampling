"""Exploratory (not pre-registered): selection bias of the smooth cells at k = 24, 36."""
import numpy as np
import plwalk as P
for k in (24, 36):
    print(f"k {k}")
    for m in P.MODELS:
        z = np.load(f"cells_{m}.npz"); row = []
        tot = {"CA_med": 0, "CA_mean": 0, "sm_r0": 0, "all_r0": 0, "sm_r1_vs_r0": []}
        for j in range(5):
            M = z[f"M_{j}"]; r0, r1, sm = z[f"r0_{j}_{k}"], z[f"r1_{j}_{k}"], z[f"sm_{j}_{k}"]
            if sm.any():
                tot["CA_med"] += M * np.median(r1[sm]); tot["CA_mean"] += M * r1[sm].mean()
                tot["sm_r0"] += M * np.median(r0[sm]); tot["sm_r1_vs_r0"].append(np.median(r1[sm] / np.maximum(r0[sm], 1e-300)))
            tot["all_r0"] += M * np.median(r0)
            row.append(f"s{j} n_sm {sm.sum():3d} M {M:.3f} med r1|sm {np.median(r1[sm]) if sm.any() else np.nan:.2e} med r0|sm {np.median(r0[sm]) if sm.any() else np.nan:.2e} med r0|all {np.median(r0):.2e} med r1|nonsm {np.median(r1[~sm]):.2e}")
        print(f" {m}: W1_CA median-based {tot['CA_med']:.2e} mean-based {tot['CA_mean']:.2e} | sum M med r0: smooth {tot['sm_r0']:.2e} all {tot['all_r0']:.2e} | r1/r0 in smooth cells (median) {np.round(tot['sm_r1_vs_r0'], 2)}")
        for r in row:
            print("    " + r)
print("k 36: smooth-cell median r1/r0 applied to all cells' mean r0, ratio at 1e6")
for m in P.MODELS:
    z = np.load(f"cells_{m}.npz")
    W = sum(z[f"M_{j}"] * np.median(z[f"r1_{j}_36"][z[f"sm_{j}_36"]] / np.maximum(z[f"r0_{j}_36"][z[f"sm_{j}_36"]], 1e-300))
            * z[f"r0_{j}_36"].mean() for j in range(5))
    print(f" {m}: {W:.2e} -> {W / 2 / 2e-4:.1f}x")
