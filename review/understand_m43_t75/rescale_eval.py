"""T84 item 4c: function-preserving positive per-neuron rescalings (Wi_k, Wh[k,:] * c_k; Wh[:,k], Wo[:,k] / c_k)
applied to each fresh noisy copy (sigma 0.05, noise seed 30, 256 copies per zoo seed; truths unchanged, taken from
fresh_0.05_30-31-32-33.json). The reviewed est75 (frozen RECIPE) is re-run; MSE ratio with its own charge.
usage: OMP_NUM_THREADS=1 python rescale_eval.py  -> rescale_eval.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import est75 as E75
from m43 import Ops
from mine import zoo_w, noisy_w
from adversarial import rescale

d = json.load(open(os.path.join(HERE, "fresh_0.05_30-31-32-33.json")))
g = np.random.default_rng(841)          # draws the reparametrisations only
out = {}
for zs in range(5):
    tr = np.array(d["zoo"][str(zs)]["truths"][:256]); ms = noisy_w(zoo_w(zs), 256, 0.05, 30)
    row = {}
    for amp in (0.0, 1.0, 2.0):
        e, ops = [], []
        for m, t in zip(ms, tr):
            o = Ops(); e.append(E75.est75(rescale(m, g.uniform(-amp, amp, 4)), o, **E75.RECIPE) - t); ops.append(o.n)
        e = np.array(e); om = np.mean(ops)
        row[amp] = dict(rms=float(np.sqrt(np.mean(e ** 2))), max=float(np.abs(e).max()), ops=float(om),
                        ratio=float(np.mean(e ** 2) / (np.mean(tr * (1 - tr)) / (om / 72))))
    out[zs] = row
    print(f"zoo {zs}: " + " | ".join(f"log c ~ U(-{a:g},{a:g}): rms {100*r['rms']:.2f}% max {100*r['max']:.1f}% ratio {r['ratio']:.4f}"
                                     for a, r in row.items()), flush=True)
json.dump(out, open(os.path.join(HERE, "rescale_eval.json"), "w"), indent=1)
