"""T424 truth cross-check: truth.py's conditional MC vs plain unconditional MC of E1(t0) on the same model, at a common
threshold. usage: python crosscheck.py TAG T0 N_PLAIN -> results/crosscheck_TAG@T0.json"""
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
    tag, t0, N = sys.argv[1], float(sys.argv[2]), int(float(sys.argv[3]))
    row = M.by_tag(tag); Wi, Wh, Wo = M.load(row); T = row["T"]
    mod = tuple(torch.from_numpy(np.ascontiguousarray(w)) for w in (Wi[:, None], Wh, Wo)); hits = 0
    for c in range(N // 200_000):
        x = np.random.default_rng([4240, c]).standard_normal((200_000, T)); o = np.argsort(x, 1)
        xs = np.take_along_axis(x, o, 1); ev = (xs[:, -2] - xs[:, -3]) > t0
        pk = np.concatenate([TR.pick(mod, torch.from_numpy(np.ascontiguousarray(x[a:a + 8192].T))) for a in range(0, 200_000, 8192)])
        hits += int((ev & (pk == o[:, -3])).sum())
    TR.HITS_TARGET = 2000; st = TR.run(tag, t0)
    pp = hits / N; z = (st["p"] - pp) / np.sqrt(pp / N + st["p"] ** 2 / st["hits"])
    out = dict(tag=tag, t0=t0, plain=dict(n=N, hits=hits, p=pp), cond=dict(n=st["n"], hits=st["hits"], p=st["p"]), z=float(z))
    print(out); json.dump(out, open(os.path.join(HERE, "results", f"crosscheck_{tag}@{t0:g}.json"), "w"), indent=1)
