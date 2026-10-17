"""T113 item 2a: sensitivity of RECIPE112's frozen knobs (eps of the sign-change rule, the Lipschitz trigger factor,
tie/clip/gate/arc tolerances), with the independent re-implementation rec112 (identical to est75 at the frozen
values). Models: clean + 16 fresh copies (abs sigma 0.05, noise seed 86) per zoo seed per rung. Ops: automatic count
(rec112 on Num) for the recipe AND for sampling (sample_ops), so every variant is priced the same way.
usage: OMP_NUM_THREADS=1 python sens.py H  -> out/sens_H<H>.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import exact as EX
import mine as M
import rec112 as R


def V(**kw):
    return type("V", (R.P,), kw)


VARIANTS = {"frozen": V(), "eps0.02": V(eps=0.02), "eps0.05": V(eps=0.05), "eps0.2": V(eps=0.2), "eps0.5": V(eps=0.5),
            "no-flip-rule": V(eps=1e9), "lip0.5": V(lip=0.5), "lip0.8": V(lip=0.8), "lip1.25": V(lip=1.25),
            "lip2": V(lip=2.0), "lip4": V(lip=4.0), "tol0": V(gtol=0.0, ctol=0.0, tie=0.0), "tol1e-9": V(gtol=1e-9, ctol=1e-9),
            "tol1e-6": V(gtol=1e-6, ctol=1e-6, tie=1e-6), "atol1e-6": V(atol=1e-6), "atol1e-12": V(atol=1e-12)}


def main():
    H = int(sys.argv[1]); so = R.sample_ops(H); t0 = time.time()
    out = {}
    for zs in range(5):
        b = M.zoo_w(H, zs); ms = [b] + M.noisy_abs(b, 16, 0.05, 86)
        tr = np.array([EX.exact_acc(m) for m in ms])
        for name, pr in VARIANTS.items():
            ests, ops = zip(*[R.counted(m, pr) for m in ms])
            e = np.array(ests) - tr; om = float(np.mean(ops[1:]))
            ratio = float(np.mean(e[1:] ** 2) / (np.mean(tr[1:] * (1 - tr[1:])) / (om / so)))
            out.setdefault(name, {})[zs] = dict(clean_err=float(e[0]), ratio=ratio, max_abs=float(np.abs(e[1:]).max()), ops=om)
        print(f"H={H} zoo {zs}: " + "; ".join(f"{k} {out[k][zs]['ratio']:.3f}" for k in VARIANTS) + f" ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(HERE, "out", f"sens_H{H}.json"), "w"))


if __name__ == "__main__":
    main()
