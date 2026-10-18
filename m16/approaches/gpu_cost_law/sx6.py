"""T270 diagnosis: per-simplex masses of the worst regions of subtree POS vs MC (which d = 6 rule fails?).
For the regions with the largest |z| in results/prefix6/mc6_POS.json: every simplex (unit rays R), |det R|, rule used
by gmass (thin: |det| < 1e-2 -> sliver split + ray rule; wide: nested Plackett), gmass mass, and an MC mass from
N ~ N(0, I_6) samples (x in cone R iff x @ inv(R) >= 0), GPU. usage: python sx6.py POS [N] [nreg]"""
import json, os, sys
import numpy as np, torch as th
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prefix6 as Q, prefix as P, gmass  # noqa: E402
pos = int(sys.argv[1]); N = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 8
nreg = int(sys.argv[3]) if len(sys.argv) > 3 else 5
w = json.load(open(os.path.join(Q.OUT, f"mc6_{pos}.json")))["worst"][:nreg]
model, front = Q.front_list(11); P._init(model, Q.K, True)
n, (hists, Rs, Is), o, s, sec = P._dfs(front[pos])
out = []
for r in w:
    R = Rs[Is == r["i"]]; dt = np.abs(np.linalg.det(R))
    m = np.array([gmass.grouped(R[j:j + 1], np.zeros(1, np.int64), 1, "cuda")[0] for j in range(len(R))])
    G = np.einsum("nij,nkj->nik", R, R); cosmin = np.array([g[~np.eye(6, dtype=bool)].min() for g in G])
    inv = th.as_tensor(np.linalg.inv(R), device="cuda")                                  # (S, 6, 6): x = lam @ R
    S = len(R); cnt = th.zeros(S, dtype=th.int64, device="cuda"); g = th.Generator(device="cuda"); g.manual_seed(77 + r["i"])
    bs = max(1 << 12, (1 << 27) // (6 * S))                                             # lam <= 1 GB
    for b in range(0, N, bs):
        x = th.randn(min(bs, N - b), 6, generator=g, device="cuda", dtype=th.float64)
        lam = th.einsum("nd,sde->nse", x, inv); cnt += (lam >= 0).all(2).sum(0)
    mc = cnt.cpu().numpy() / N
    out.append(dict(region=r["i"], mass=r["mass"], region_mc=r["mc"], simplices=[dict(det=float(dt[j]), thin=bool(dt[j] < 1e-2),
               cosmin=float(cosmin[j]), mass=float(m[j]), mc=float(mc[j])) for j in np.argsort(-mc)[:12]],
               sum_mass=float(m.sum()), sum_mc=float(mc.sum()), n_simplices=S,
               thin_mass=float(m[dt < 1e-2].sum()), thin_mc=float(mc[dt < 1e-2].sum())))
    print(json.dumps({k: v for k, v in out[-1].items() if k != "simplices"}), flush=True)
    for x in out[-1]["simplices"][:5]:
        print("   ", x, flush=True)
json.dump(out, open(os.path.join(Q.OUT, f"sx6_{pos}.json"), "w"), indent=1)
