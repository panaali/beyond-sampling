"""T101 mechanism tracker across training amount (exact enumeration used only to DESCRIBE where errors are; it is not
part of any estimator). For each k and zoo seed, on the clean model:
  * accuracy, weight scale (|Wi|, ||Wh||, ||Wo||);
  * step-2 switch ratios r = x1/x0 at which each neuron k switches at step 2, per chart sign(x0):
    h2_k = |x0| (Wh relu(Wi sgn))_k + Wi_k x1 = 0  ->  r_k = -sgn (Wh relu(Wi sgn))_k / Wi_k;
    'switch near x1 = x0' = min_k |r_k - 1| per chart (and the neuron);
  * step-1 storage: which neurons hold x0 in each chart (relu(Wi sgn) > 0);
  * error mass by number of active step-3 neurons (diag75.describe), share of all-off ('dead', tie -> class 0) error,
    share touching a true boundary, mean number of step-3 gates cutting a step-2 piece.
usage: python mech.py k [k ...]  -> mech_k{k}.json and printed table"""
import json
import os
import sys

import numpy as np

import lt
from diag75 import describe

HERE = os.path.dirname(os.path.abspath(__file__))


def mech(m):
    Wi, Wh, Wo = m
    out = dict(Wi=Wi.tolist(), Wi_abs_max=float(np.abs(Wi).max()), Wh_norm=float(np.linalg.norm(Wh, 2)),
               Wo_norm=float(np.linalg.norm(Wo, 2)), charts={})
    for sgn in (1, -1):
        g1 = np.maximum(Wi * sgn, 0)
        c = Wh @ g1
        with np.errstate(divide="ignore"):
            r = -sgn * c / Wi
        j = int(np.argmin(np.abs(r - 1)))
        out["charts"][sgn] = dict(store=[int(x) for x in np.nonzero(g1 > 0)[0]], r=r.tolist(), near1=float(abs(r[j] - 1)),
                                  near1_neuron=j, near1_r=float(r[j]))
    by_n, touch, cr = describe(m)
    tot = sum(by_n.values())
    out.update(err=float(tot), err_by_active={int(a): float(v / tot) for a, v in sorted(by_n.items())},
               dead_share=float(by_n.get(0, 0) / tot), touch_share=float(touch[True] / tot), gates_cut=float(cr))
    return out


def main():
    for k in map(int, sys.argv[1:]):
        res = {}
        for s in range(5):
            m = lt.zoo(k, s); d = mech(m); d["acc"] = float(lt.truth(m)[0]); d["regions"] = lt.regions(m); res[s] = d
            ch = d["charts"]
            print(f"k={k} s={s} acc {d['acc']:.4f} reg {d['regions']} |Wi|max {d['Wi_abs_max']:.1f} ||Wh|| {d['Wh_norm']:.2f} "
                  f"||Wo|| {d['Wo_norm']:.1f} | x0>0 store {ch[1]['store']} r {np.round(ch[1]['r'], 2).tolist()} | "
                  f"x0<0 store {ch[-1]['store']} r {np.round(ch[-1]['r'], 2).tolist()} | err by #on "
                  + ",".join(f"{a}:{v:.0%}" for a, v in d["err_by_active"].items()) + f" touch {d['touch_share']:.0%} cut {d['gates_cut']:.2f}",
                  flush=True)
        json.dump(res, open(os.path.join(HERE, f"mech_k{k}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
