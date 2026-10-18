"""T279 at (16,4) zoo seed 0: certified mass vs cones (CROWN and IBP), gap>=0.1 diagnostic, estimate and bound."""
import json
import sys
import time
import numpy as np
from tree import Tree, load, forward_argmax

m = load("/home/user/dl/zoo/z16_4_0.pth")
rng = np.random.default_rng(279)
ok = 0
for _ in range(40):                                    # truth: 4e7 samples (scoring only)
    x = rng.standard_normal((1_000_000, 4))
    ok += int((forward_argmax(m, x) == np.argsort(x, 1)[:, -2]).sum())
truth = ok / 4e7
xd = np.random.default_rng(2790).standard_normal((1_000_000, 4))   # diagnostic points
s = -np.sort(-xd, 1)
gap = np.minimum(s[:, 0] - s[:, 1], s[:, 1] - s[:, 2]) >= 0.1
print("truth", truth, "gap>=0.1 mass", gap.mean(), flush=True)
out = {"truth": truth, "truth_se": float(np.sqrt(truth * (1 - truth) / 4e7)), "gap_mass_mc": float(gap.mean())}
checkpoints = [int(c) for c in sys.argv[2].split(",")]
for bound in sys.argv[1].split(","):
    tr = Tree(m, bound)
    rows = []
    t0 = time.time()
    for cp in checkpoints:
        while tr.n_leaves < cp and tr.heap:
            tr.step()
        leaf = tr.locate(xd)
        st = np.array([tr.nodes[l]["status"] for l in leaf])
        r = {"leaves": tr.n_leaves, "cert_correct": tr.mass[1], "cert_wrong": tr.mass[-1], "uncert": tr.mass[0],
             "estimate": tr.mass[1] + 0.5 * tr.mass[0], "bias": tr.mass[1] + 0.5 * tr.mass[0] - truth,
             "bound": 0.5 * tr.mass[0], "flops": tr.fl.n, "gap_cert_frac": float((st[gap] != 0).mean()),
             "uncert_blind_frac": float(((st == 0) & (s[:, 1] <= 0)).sum() / max((st == 0).sum(), 1)),
             "cpu_s": time.time() - t0}
        rows.append(r)
        print(bound, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    out[bound] = rows
json.dump(out, open(f"results/t4_{sys.argv[1].replace(',', '_')}.json", "w"), indent=1)
