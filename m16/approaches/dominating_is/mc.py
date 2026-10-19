"""Plain MC reference for P(E1(t0)) (prefiltered by g23 >= t0, as T400). usage: python mc.py t0 n seed -> mc_<t0>_<seed>.json"""
import json, sys, time
import numpy as np
from net import event
t0 = float(sys.argv[1]); n = int(float(sys.argv[2])); seed = int(sys.argv[3])
rng = np.random.default_rng(436_000 + seed); B = 2_000_000; done = kept = 0; hits = []; t = time.time()
while done < n:
    m = min(B, n - done); X = rng.standard_normal((m, 10)); hit, top, k = event(X, t0)
    kept += k; hits += [top[i].tolist() for i in np.flatnonzero(hit)]; done += m
out = dict(t0=t0, n=n, seed=seed, kept=kept, hits=len(hits), p=len(hits) / n, triples=hits, sec=time.time() - t)
json.dump(out, open(f"mc_{t0}_{seed}.json", "w"))
print(t0, n, "kept", kept, "hits", len(hits), "p %.3e" % (len(hits) / n), "sec %.0f" % out["sec"], flush=True)
