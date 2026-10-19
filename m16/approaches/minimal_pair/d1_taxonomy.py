"""D1: error taxonomy by plain sampling. For each model, misclassified inputs by (pred rank, positions)."""
import json, sys
import numpy as np
import core as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
out = {}
for T in (4, 5):
    for s in range(5):
        m = C.zoo(16, T, s); rng = np.random.default_rng(100 + 10 * T + s)
        rows = []
        for _ in range(N // 250_000):
            x = rng.standard_normal((250_000, T))
            pred = C.forward(m, x).argmax(1)
            order = np.argsort(-x, 1)                  # order[:,r] = position of rank r+1
            rank_of = np.argsort(order, 1)             # rank_of[i,p] = rank-1 of position p
            pr = rank_of[np.arange(len(x)), pred] + 1
            bad = pr != 2
            xs = -np.sort(-x, 1)
            for i in np.nonzero(bad)[0]:
                rows.append((pr[i], *order[i, :3], xs[i, 0] - xs[i, 1], xs[i, 1] - xs[i, 2]))
        R = np.array(rows, float)
        err = len(R) / N
        d = {"err": err, "n_err": len(R)}
        d["pred_rank"] = {int(k): float((R[:, 0] == k).mean()) for k in range(1, T + 1) if k != 2}
        # which gap is small: for pred rank 1 relevant gap g12, rank 3 -> g23
        g = np.where(R[:, 0] == 1, R[:, 4], np.where(R[:, 0] == 3, R[:, 5], np.nan))
        d["far_share"] = float(np.nanmean(g > 0.3)) if np.isfinite(g).any() else None
        d["gap_q"] = {q: float(np.nanquantile(g, q)) for q in (0.5, 0.9, 0.99)}
        # position of rank-2 element and rank-1 element
        d["pos_r1"] = np.bincount(R[:, 1].astype(int), minlength=T).tolist()
        d["pos_r2"] = np.bincount(R[:, 2].astype(int), minlength=T).tolist()
        d["pos_r3"] = np.bincount(R[:, 3].astype(int), minlength=T).tolist()
        # top (r1,r2,r3) position triples
        trip = {}
        for r in R:
            k = f"{int(r[0])}:{int(r[1])}{int(r[2])}{int(r[3])}"; trip[k] = trip.get(k, 0) + 1
        d["top"] = sorted(((v / N, k) for k, v in trip.items()), reverse=True)[:15]
        out[f"{T}_{s}"] = d
        print(T, s, f"err={err:.5f}", d["pred_rank"], "far", round(d["far_share"], 3), d["gap_q"], flush=True)
        print("   r1pos", d["pos_r1"], "r2pos", d["pos_r2"], "r3pos", d["pos_r3"])
        print("   top", [(f"{v*1e4:.1f}e-4", k) for v, k in d["top"][:8]], flush=True)
json.dump(out, open("d1_taxonomy.json", "w"), indent=1)
