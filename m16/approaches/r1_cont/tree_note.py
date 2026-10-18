"""T326 note for T329: where the B2 prefix-tree floor comes from. Per line and step: distinct prefix states (model-free:
they depend only on the line inputs); the error mass per node (sum over the rows through the node of the row's
deficit on that line, as in floors.py); the nodes carrying > 1e-4 of it; and the all-low prefixes.
Analysis of the existing T317e walks on the 80 DEV2 models; no new knobs. -> results_tree_note.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
from cont_e import T, NT, TMAX  # noqa: E402


def prefix_ids(x0, d):
    """Per step s, the node id of each row's prefix (positions 0..s) and the node count."""
    inv_prev = np.zeros(len(x0), int); ids = []
    for s in range(T):
        key = np.round(np.column_stack([inv_prev, x0[:, s], d[:, s]]), 12)
        _, inv = np.unique(key, axis=0, return_inverse=True); inv_prev = inv.ravel(); ids.append(inv_prev.copy())
    return ids


def labels():
    """Model-free role of each position per triple: 1, 2, 3 (top-3), 4 (v4 at P4), L (low)."""
    lab = np.full((NT, T), "L")
    for j, c in enumerate("123"):
        lab[np.arange(NT), E.TRI[:, j]] = c
    lab[np.arange(NT), E.P4] = "4"
    return lab


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
    import r0  # noqa: E402
    lab = labels(); out = {}
    lines = {f"{tb['name']}-{kind}": (tb, kind) for tb in E.TABLES for kind in ("23", "12") if f"line{kind}" in tb}
    ids = {l: prefix_ids(*tb[f"line{kind}"]) for l, (tb, kind) in lines.items()}
    for l in lines:
        per_step = [int(i.max() + 1) for i in ids[l]]
        role_step = [len({"".join(r[:s + 1]) for r in lab}) for s in range(T)]
        alllow = [int(any(set(r[:s + 1]) == {"L"} for r in lab)) for s in range(T)]
        out[l] = dict(nodes_per_step=per_step, total=sum(per_step), role_strings_per_step=role_step, all_low_nodes=sum(alllow))
    # error mass per node (per model), using floors.py's per-row deficit
    import floors as FL  # noqa: E402
    thr = (1e-4, 1e-5, 1e-6); cnt = {l: {t: [] for t in thr} for l in lines}; share = {l: [] for l in lines}; drop_mass = {}; kept_nodes = {}
    for nm, m in r0.dev2():
        model = tuple(np.asarray(w, np.float64) for w in m); k = E.TRI[:, 1]; I = {}
        for l, (tb, kind) in lines.items():
            x, dx = tb[f"line{kind}"]; p, _, _ = FL.walk_rec(model, x, dx, k, TMAX)
            R, A, B, _ = E.merge(p, NT, TMAX); I[l] = E.profile(R, A, B, NT, tb["nodes"], kind, TMAX, tb["phi0"])[0]
        for l, (tb, kind) in lines.items():
            nm_ = tb["name"]; full = tb["phi0"]["m8"][None] if kind == "23" else tb["blind12"][None]
            other = I.get(f"{nm_}-12", tb["blind12"][None]) if kind == "23" else I[f"{nm_}-23"]
            dfc = np.maximum(((((full - I[l]) * other * tb["dens"][None]) @ tb["wts"]) / NT), 0.0)
            node_mass = np.concatenate([np.bincount(i, weights=dfc) for i in ids[l]])
            for t in thr:
                cnt[l][t].append(int((node_mass > t).sum()))
                keep = dfc > t; drop_mass.setdefault((l, t), []).append(float(dfc[~keep].sum()))
                kn = sum(len(np.unique(i[keep])) for i in ids[l]); kept_nodes.setdefault((l, t), []).append(kn)
            share[l].append(float(dfc.sum()))
    for l in lines:
        out[l]["nodes_mass_gt"] = {str(t): float(np.mean(cnt[l][t])) for t in thr}
        out[l]["prune_rows_below"] = {str(t): dict(dropped_deficit_pp=100 * float(np.mean(drop_mass[(l, t)])),
                                                   tree_nodes=float(np.mean(kept_nodes[(l, t)]))) for t in thr}
        out[l]["mean_line_deficit_pp"] = 100 * float(np.mean(share[l]))
    json.dump(out, open(os.path.join(HERE, "results_tree_note.json"), "w"), indent=1)
    for l, v in out.items():
        print(l, v)
