"""T206: estT.orthant op count per simplicial cone by dimension d (counter only; random cones of varied thinness,
rng 3). -> orthbench.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "arc_m43_repro")]
import numpy as np, estT
from m43 import Ops
rng = np.random.default_rng(3); out = {}
for d in (3, 4, 5):
    ops = []
    for sc in (1.0, 0.1, 0.01, 0.001):
        for _ in range(30):
            c = rng.standard_normal(d); R = c + sc * rng.standard_normal((d, d)); R /= np.linalg.norm(R, axis=1)[:, None]
            if np.linalg.det(R) < 0: R = R[[1, 0] + list(range(2, d))]
            o = Ops(); estT.orthant(R, o); ops.append(o.n)
    out[d] = dict(min=int(min(ops)), median=float(np.median(ops)), max=int(max(ops)))
    print(d, out[d])
json.dump(out, open(os.path.join(HERE, "orthbench.json"), "w"), indent=1)
