"""T350: all variants (A, B) for all events on a weights npz; writes estimates only (no truths read)."""
import json
import sys
import numpy as np
import rare

path, out, idx = sys.argv[1], sys.argv[2], (sys.argv[3:] or None)
z = np.load(path, allow_pickle=True); names = [str(n) for n in z["names"]]
res = {}
for i, nm in enumerate(names):
    if idx and nm not in idx and str(i) not in idx:
        continue
    m = tuple(np.asarray(z[k][i], float) for k in ("Wi", "Wh", "Wo"))
    a1, ca1, a2, ca2 = rare.est_A(m); b1, cb1, b2, cb2 = rare.est_B(m)
    res[nm] = {"A": {"E1": {str(k): float(v) for k, v in a1.items()}, "C_E1": float(ca1),
                     "E2": {str(k): float(v) for k, v in a2.items()}, "C_E2": float(ca2)},
               "B": {"E1": {str(k): float(v) for k, v in b1.items()}, "C_E1": float(cb1),
                     "E2": {str(k): float(v) for k, v in b2.items()}, "C_E2": float(cb2)}}
    print(nm, json.dumps(res[nm]), flush=True)
json.dump(res, open(out, "w"), indent=1)
