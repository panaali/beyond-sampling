"""T84 item 2: selection risk of the threshold eps = 0.1 and of the impact definition. Runs the independent
re-implementation (recipe.py) for every variant on (a) T75's dev set (noise seed 0, 32 copies per zoo seed, where
eps / impact were chosen) and (b) fresh noise seed 30 (first 128 copies per zoo seed). Ops: recipe.py's charge
(T75 strict convention, re-applied). Truth: exact.py.
usage: OMP_NUM_THREADS=1 python sweep.py  -> sweep.json, sweep.log"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import exact as EX
import recipe as R
from mine import zoo_w, noisy_w

EPS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
VARIANTS = [("T75 (depth x max|dW|, scale max)", dict(wmode="max", smode="max")),
            ("depth only", dict(wmode="one", smode="max")),
            ("depth x L2|dW|", dict(wmode="l2", smode="max")),
            ("scale = vertex mean", dict(wmode="max", smode="mean"))]
SETS = {"dev seed 0 (n=32)": (0, 32), "fresh seed 30 (n=128)": (30, 128)}


def run(models, truths, **kw):
    e, ops = [], []
    for m, t in zip(models, truths):
        ch = R.Charge(); e.append(R.recipe(m, ch, **kw) - t); ops.append(ch.n)
    e = np.array(e); om = float(np.mean(ops))
    return float(np.mean(e ** 2) / (np.mean(truths * (1 - truths)) / (om / 72))), om, float(np.sqrt(np.mean(e ** 2)))


def main():
    t0 = time.time(); data = {}
    for sname, (ns, n) in SETS.items():
        for zs in range(5):
            ms = noisy_w(zoo_w(zs), n, 0.05, ns)
            data[(sname, zs)] = ([R.as_lists(m) for m in ms], np.array([EX.exact_acc(m) for m in ms]))
    out = []
    jobs = [(f"{v} eps={e}", dict(kw, eps=e)) for v, kw in VARIANTS for e in EPS]
    jobs += [(f"margin-aware step 3, eps={e}", dict(eps=(0.1, e), mmode="margin")) for e in (0.1, 0.3, 1.0)]
    jobs += [("seed-4 probe: eps=(0.1, 0.03)", dict(eps=(0.1, 0.03))), ("seed-4 probe: eps=(0.03, 0.1)", dict(eps=(0.03, 0.1))),
             ("exact limit eps=0", dict(eps=0.0))]
    for name, kw in jobs:
        row = dict(name=name, kw={k: v for k, v in kw.items()})
        for sname in SETS:
            rs = [run(*data[(sname, zs)], **kw) for zs in range(5)]
            row[sname] = rs
            print(f"{name:42s} {sname:22s} ratios " + " ".join(f"{r[0]:.4f}" for r in rs)
                  + f" | max {max(r[0] for r in rs):.4f} | ops " + " ".join(f"{r[1]/1000:.1f}k" for r in rs), flush=True)
        out.append(row)
    json.dump(out, open(os.path.join(HERE, "sweep.json"), "w"), indent=1, default=str)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
