"""T368: frozen T360 K3 (rare_k.est) on the 10 deep-tail pairs; scores vs T192 conditional-law truths."""
import hashlib
import json
import sys
import numpy as np

H = {"/home/user/clean-neurons/m16/approaches/rare_k/rare_k.py": "89d073825fa21433355aaad63e95e7bbba63676d9c6f4cdd421af9298f55196f",
     "/home/user/clean-neurons/m16/approaches/rare_check/rare.py": "e7db88752c8c1139793eb23994017a8685a36003dc03553103d54d8a9bc06b97",
     "/home/user/clean-neurons/m16/approaches/r1w_repro/r1w_indep.py": "f737102b57775a7c28729ad066fa1170e7128cbd97e3fbf5629465d41507df3d",
     "/home/user/clean-neurons/m16/approaches/rare_check/cont_e.py": "50dd1c9b85512e23f037171d19de03527f8f5989758cfb74d93b441de48fda2b"}
for f, h in H.items():
    assert hashlib.sha256(open(f, "rb").read()).hexdigest() == h, f
sys.path.insert(0, "/home/user/clean-neurons/m16/approaches/rare_k")
import rare_k as RK  # noqa: E402

JOBS = [("t364", "dev3_s526", (3.25, 3.5))] + [("t366", f"dev3_s{s}", (2.35, 2.75)) for s in range(528, 532)]
out = {}
for d, nm, t0s in JOBS:
    z = np.load(f"/home/user/dl/t368/{d}_w.npz", allow_pickle=True); i = [str(n) for n in z["names"]].index(nm)
    m = tuple(np.asarray(z[k][i], float) for k in ("Wi", "Wh", "Wo"))
    e1, c1, _, _ = RK.est(m, 3, t0s=t0s, ts=())
    tr = json.load(open(f"/home/user/dl/t368/{d}_tail.json"))
    for t0 in t0s:
        key = f"{nm}@{t0:g}"; p = tr[key]["p"]
        out[key] = dict(est=float(e1[t0]), truth=p, truth_se=tr[key]["se"], ratio=float(e1[t0]) / p, C_fpe=float(c1),
                        within2x=bool(.5 <= e1[t0] / p <= 2), within1p3=bool(1 / 1.3 <= e1[t0] / p <= 1.3))
        print(key, out[key], flush=True)
out["_summary"] = {"within2x": sum(v["within2x"] for k, v in out.items() if not k.startswith("_")),
                   "within1.3x": sum(v["within1p3"] for k, v in out.items() if not k.startswith("_")), "n": 10}
json.dump(out, open("results/deep.json", "w"), indent=1); print(out["_summary"])
