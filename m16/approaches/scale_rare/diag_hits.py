"""T424 post-hoc diagnostic (not in the pre-registered score): replay truth.py's draws (same per-chunk seeds) for one
point and record each hit's g12, g23, v2 (to locate the error mass the walks miss).
usage: python diag_hits.py TAG T0 -> results/diag_hits_TAG@T0.json"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import models as M  # noqa: E402
import truth as TR  # noqa: E402

if __name__ == "__main__":
    torch.set_num_threads(1)
    tag, t0 = sys.argv[1], float(sys.argv[2])
    st = json.load(open(os.path.join(HERE, "results", "truth", f"{tag}@{t0:g}.json")))
    row = M.by_tag(tag); Wi, Wh, Wo = M.load(row); T = row["T"]
    mod = tuple(torch.from_numpy(np.ascontiguousarray(w)) for w in (Wi[:, None], Wh, Wo)); draw = TR.make_draw(T, t0); out = []
    for c in range(st["chunks"]):
        x, p3, g23, g12 = draw(np.random.default_rng(st["seed"] + [c]), TR.CHUNK); xT = torch.from_numpy(np.ascontiguousarray(x.T))
        hit = np.concatenate([TR.pick(mod, xT[:, a:a + TR.BLK]) for a in range(0, TR.CHUNK, TR.BLK)]) == p3
        v2 = np.sort(x, 1)[:, -2]
        out += [dict(g12=float(a), g23=float(b), v2=float(v)) for a, b, v in zip(g12[hit], g23[hit], v2[hit])]
    assert len(out) == st["hits"], (len(out), st["hits"])
    g = np.array([o["g12"] for o in out])
    summ = dict(tag=tag, t0=t0, hits=len(out), g12_quantiles={q: float(np.quantile(g, q)) for q in (.1, .25, .5, .75, .9)},
                frac_g12_lt={str(a): float((g < a).mean()) for a in (1e-3, 3e-3, 1e-2, .03, .1)})
    print(summ); json.dump(dict(summary=summ, hits=out), open(os.path.join(HERE, "results", f"diag_hits_{tag}@{t0:g}.json"), "w"), indent=1)
