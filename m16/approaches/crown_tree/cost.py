"""T279 cost breakdown: certified mass and FLOPs vs cones for crown (full back-substitution), crown_fwd (shared
forward intermediate bounds + final back-substitution) and ibp, at (16,4) zoo seed 0. No diagnostic sampling."""
import json, sys, time
from tree import Tree, load
m = load("/home/user/dl/zoo/z16_4_0.pth")
out = {}
for b in ("crown", "crown_fwd", "ibp"):
    tr = Tree(m, b); t0 = time.time(); rows = []
    for cp in (1000, 3000, 10000, 30000, 100000):
        while tr.n_leaves < cp and tr.heap:
            tr.step()
        r = dict(leaves=tr.n_leaves, certified=float(tr.mass[1] + tr.mass[-1]), cert_wrong=float(tr.mass[-1]),
                 uncert=float(tr.mass[0]), flops=tr.fl.n, flops_bound=tr.flops_bound, flops_mass=tr.flops_mass,
                 flops_per_cone=tr.fl.n / len(tr.nodes), cpu_s=time.time() - t0)
        rows.append(r)
        print(b, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    out[b] = rows
json.dump(out, open("results/cost.json", "w"), indent=1)
