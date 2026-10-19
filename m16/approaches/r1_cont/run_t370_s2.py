"""T370 stage 2: frozen Q16 (t370 @6441c0c5, hash-checked against t370_q16_freeze_sha256.txt) on all 64 candidates,
truth-blind: dev3_s548-563 x {E1 at t0 2.35, 2.75; E2 at t 1.25, 1.5}. usage: python run_t370_s2.py e1|e2
-> results_t370_s2_<part>.json (resumes; T370_ONE=1 runs one job per call)."""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(HERE, "t370_q16_freeze_sha256.txt")):
    h, f = line.split()
    p = os.path.join(HERE, f) if os.path.exists(os.path.join(HERE, f)) else os.path.join(os.environ["T340_SRC"], "m16", "approaches", f)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, f"hash mismatch: {f}"
sys.path.insert(0, HERE)
import t370 as Z  # noqa: E402

M = 16
MODELS = [f"dev3_s{i}" for i in range(548, 564)]
MODELS_E1 = MODELS[12:]                                  # split with T286 (22:20Z): it runs Q16 E1 on s548-559
E1T = (2.35, 2.75)
E2T = (1.25, 1.5)

if __name__ == "__main__":
    part = sys.argv[1]
    z = np.load(os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")); names = [str(s) for s in z["names"]]
    mod = lambda nm: tuple(z[k][names.index(nm)].astype(float) for k in ("Wi", "Wh", "Wo"))
    fn = os.path.join(HERE, f"results_t370_s2_{part}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    jobs = [(nm, None) for nm in MODELS_E1] if part == "e1" else [(nm, t) for t in E2T for nm in MODELS]
    done = 0
    for nm, t in jobs:
        key = nm if t is None else f"{nm}|{t}"
        if key in out:
            continue
        t0 = time.time()
        if t is None:
            e, f = Z.est_e1(mod(nm), E1T, M); rec = dict(est={str(k): v for k, v in e.items()})
        else:
            e, f = Z.est_e2(mod(nm), t, M); rec = dict(est=e)
        rec.update(C=f / 5948, sec=time.time() - t0); out[key] = rec; json.dump(out, open(fn, "w"), indent=1)
        print(key, rec, flush=True)
        done += 1
        if os.environ.get("T370_ONE") or done >= int(os.environ.get("T370_N", "1000000")):        # jobs per call
            break
