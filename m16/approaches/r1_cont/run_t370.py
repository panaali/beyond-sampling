"""T370 stage 1 runner: variant M (V1 = 16, V2 = 32, V3 = 64). usage: python run_t370.py VARIANT e1|e2 (S14 | Q16 | Q32) -> results_t370_<VARIANT>_<part>.json
E1: s547 (2.25, 2.5) + controls s541 (1.75, 2.0), s526 (3.25, 3.5), s528-s531 (2.35, 2.75). E2: T363's 8 models x 4 t."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t370 as Z  # noqa: E402

E1 = {"dev3_s547": (2.25, 2.5), "dev3_s541": (1.75, 2.0), "dev3_s526": (3.25, 3.5), "dev3_s528": (2.35, 2.75),
      "dev3_s529": (2.35, 2.75), "dev3_s530": (2.35, 2.75), "dev3_s531": (2.35, 2.75)}
E2M = ["dev3_s508", "dev3_s511", "dev3_s512", "dev3_s515", "dev3_s517", "dev3_s518", "dev3_s519", "dev3_s525"]
E2T = (0.5, 0.75, 1.0, 1.25)

if __name__ == "__main__":
    var, part = sys.argv[1], sys.argv[2]; M, cl = (None, "S14") if var == "S14" else (int(var[1:]), None)
    z = np.load(os.path.join(HERE, "..", "..", "..", "data", "dev3", "weights.npz")); names = [str(s) for s in z["names"]]
    mod = lambda nm: (z["Wi"][names.index(nm)].astype(float), z["Wh"][names.index(nm)].astype(float), z["Wo"][names.index(nm)].astype(float))
    fn = os.path.join(HERE, f"results_t370_{var}_{part}.json"); out = json.load(open(fn)) if os.path.exists(fn) else {}
    jobs = [(nm, None) for nm in E1] if part == "e1" else [(nm, t) for nm in E2M for t in E2T]
    for nm, t in jobs:
        key = nm if t is None else f"{nm}|{t}"
        if key in out:
            continue
        t0 = time.time()
        if t is None:
            e, f = Z.est_e1(mod(nm), E1[nm], M, closure=cl); rec = dict(est={str(k): v for k, v in e.items()})
        else:
            e, f = Z.est_e2(mod(nm), t, M, closure=cl); rec = dict(est=e)
        rec.update(C=f / 5948, sec=time.time() - t0); out[key] = rec; json.dump(out, open(fn, "w"), indent=1)
        print(key, rec, flush=True)
