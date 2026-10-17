"""T113 item 4a: automatic op recount. rec112 on the Num type counts every executed + - * / abs, comparison, sqrt
and atan2 on model-dependent numbers; the same counter prices one sampling step (3 RNN steps, ReLU, readout, argmax,
label, compare). Compared with est75's own strict charge and the 3H(H+1)+3H sampling convention used by T112.
usage: OMP_NUM_THREADS=1 python count.py  -> out/count.log, out/count.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import est75 as E75
from m43 import Ops
import mine as M
import rec112 as R

out = {}
for H in (4, 5, 6, 8, 16):
    so = R.sample_ops(H); conv = 3 * H * (H + 1) + 3 * H; rows = []
    for zs in range(5):
        b = M.zoo_w(H, zs)
        for m in [b] + M.noisy_abs(b, 4, 0.05, 89):
            o = Ops(); e = E75.est75(m, o, **E75.RECIPE112); r, n = R.counted(m)
            rows.append(dict(zoo=zs, charge=o.n, auto=n, diff=r - e))
    a = np.array([q["auto"] for q in rows]); c = np.array([q["charge"] for q in rows])
    mult = float(np.mean(a / so) / np.mean(c / conv))
    out[H] = dict(sample_auto=so, sample_conv=conv, rows=rows, mult=mult)
    print(f"H={H}: recipe auto FLOPs / est75 charge {np.mean(a/c):.3f} ({np.min(a/c):.3f}-{np.max(a/c):.3f}); sample auto {so} vs "
          f"convention {conv}; recipe in sample units: auto {np.mean(a)/so:.0f} vs charged {np.mean(c)/conv:.0f} -> MSE-ratio "
          f"multiplier x{mult:.2f}; max |rec112 - est75| {max(abs(q['diff']) for q in rows):.1e}", flush=True)
json.dump(out, open(os.path.join(HERE, "out", "count.json"), "w"))
