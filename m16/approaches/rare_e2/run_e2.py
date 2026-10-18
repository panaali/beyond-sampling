"""T358: frozen variant A (../rare_check/rare.py est_E2) on dev3_s506..s525; estimates only, no truths."""
import hashlib
import json
import sys
import numpy as np

sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/rare_check")
SRC = "/home/user/clean-neurons/m16/approaches/rare_check/rare.py"
assert hashlib.sha256(open(SRC, "rb").read()).hexdigest() == "e7db88752c8c1139793eb23994017a8685a36003dc03553103d54d8a9bc06b97"
import rare  # noqa: E402

TS = (0.5, 0.75, 1.0, 1.25)
z = np.load("/home/user/dl/dev3/weights.npz", allow_pickle=True); names = [str(n) for n in z["names"]]
want = [f"dev3_s{i}" for i in range(506, 526)]
res = {}
for nm in want:
    i = names.index(nm); m = tuple(np.asarray(z[k][i], float) for k in ("Wi", "Wh", "Wo"))
    e, c = rare.est_E2(m, TS)
    res[nm] = {"E2": {str(t): float(v) for t, v in e.items()}, "C_E2": float(c)}
    print(nm, res[nm], flush=True)
json.dump(res, open("results/est.json", "w"), indent=1)
