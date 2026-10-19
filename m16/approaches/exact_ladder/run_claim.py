"""python run_claim.py T name...  -> like run_list.py, but skips models already in est_info.jsonl and claims each model
with an O_EXCL file in claims/ so that two workers never compute the same model."""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.append("../..")
import leaderboard as L  # noqa: E402
import estimator  # noqa: E402

T = int(sys.argv[1])
ms = L.models(16, T)
os.makedirs("claims", exist_ok=True)
for name in sys.argv[2:]:
    key = hashlib.sha1(b"".join(np.ascontiguousarray(W).tobytes() for W in ms[name])).hexdigest()[:12]
    if any(json.loads(l).get("key") == key for l in open("est_info.jsonl")):
        continue
    try:
        os.close(os.open(f"claims/T{T}_{name}", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        continue
    t0 = time.time()
    a, c = estimator.est(ms[name], 0, None)
    print(f"{name} acc {a:.10f} fpe {c:.4g} wall {time.time() - t0:.0f}s", flush=True)
