"""T364 deep-tail run: the T218 child's frozen K3 estimator (T340c + walk fix, segment start = T354 est_seg; read-only
copy of claude/t218-tie-tail-labels @e433919d r1_cont/{t340c,t354,walkfix}.py), E1(t0) = {pick p3, g23 > t0} at deep t0.
Deductive (no random inputs). usage: python deep.py T_hi t0,t0,... [models] -> results/deep_<T_hi>.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R1C = os.environ.get("R1C", "/tmp/claude-0/-home-user-clean-neurons/18e70fa5-7ff5-5fac-af53-4fea2351128f/scratchpad/r1c/m16/approaches/r1_cont")
os.environ.setdefault("T340_SRC", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, R1C)
import t340c as TC  # noqa: E402
import t354 as S  # noqa: E402
import walkfix as WF  # noqa: E402

TC.rare.walk3 = lambda model, x0, d, k, k3, tmax, max_cells=5000: WF.walk3_fixed(TC.G, model, x0, d, k, k3, tmax, max_cells)[:5]


def models():
    ms = {"orig": TC.cf.lb3_models()["orig"]}
    z = np.load(os.path.join(HERE, "ext", "dev3", "weights.npz"))
    for i, n in enumerate(z["names"]):
        ms[str(n)] = (z["Wi"][i], z["Wh"][i], z["Wo"][i])
    return ms


def main():
    thi = float(sys.argv[1]); t0s = tuple(float(t) for t in sys.argv[2].split(","))
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else ["orig", "dev3_s526", "dev3_s527"]
    ms = models(); out = {}
    for nm in names:
        t = time.time(); r, r2, f = S.est_seg(tuple(np.asarray(w, float) for w in ms[nm]), 3, t0s=t0s, t2s=(), tmax=thi)
        out[nm] = dict(est={str(k): v for k, v in r.items()}, C=f / TC.cf.FPE, sec=time.time() - t)
        print(nm, {k: f"{v:.3e}" for k, v in r.items()}, f"C {f / TC.cf.FPE:.0f}", f"{time.time() - t:.0f}s", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", f"deep_{thi}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
