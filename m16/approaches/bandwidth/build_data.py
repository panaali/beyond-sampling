"""Build (surface point -> band width) datasets. python build_data.py -> data/<model>.npz"""
import sys, time
import numpy as np
import wtool as W
names = sys.argv[1:] or (["orig"] + [f"seed{s}" for s in range(5)] +
                         [f"noisy0.0025_{k}" for k in range(3)] + [f"noisy0.005_{k}" for k in range(3)])
for nm in names:
    M = W.get_model(nm)
    n = 60000 if nm == "orig" else 30000
    t0 = time.time()
    R = W.build(M, n, seed=2026)    # same surface points for every model (paired comparisons)
    np.savez_compressed(f"data/{nm}.npz", **R)
    print(nm, f"{time.time()-t0:.0f}s", "E[mass]", R["mass"].mean(), "E|delta|", np.abs(R["delta"]).mean(),
          "pieces", R["pieces"].mean(), flush=True)
