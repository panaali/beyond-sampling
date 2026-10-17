"""Ground truth for T in {3, 6} (the leaderboard caches only T=4,5,10): same models and the same estimator of
truth as m16/leaderboard.py compute_truths (4M fresh inputs per model, seeded identically), written HERE
(results_truths_H16_T{T}.json) so that no shared file is touched.  usage: python truths_small.py T [n]"""
import json, os, sys, zlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from leaderboard import models, correct

T = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000; H = 16
path = os.path.join(os.path.dirname(__file__), f"results_truths_H{H}_T{T}.json")
out = json.load(open(path)) if os.path.exists(path) else {}
for name, m in models(H, T).items():
    if name in out and out[name]["n"] >= n:
        continue
    rng = np.random.default_rng(987_654 + zlib.crc32(f"{H},{T},{name}".encode()) % 10_000)
    ok = 0
    for _ in range(n // 500_000):
        ok += int(correct(m, rng.standard_normal((500_000, T))).sum())
    p = ok / n
    out[name] = {"acc": p, "se": float(np.sqrt(p * (1 - p) / n)), "n": n}
    print(name, out[name], flush=True)
    json.dump(out, open(path, "w"), indent=1)
