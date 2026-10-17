"""T84 item 4: adversarial robustness of the frozen T75 recipe (estimates by recipe.py, identical to est75 to 1e-15;
truth exact.py).
 part 1  function-preserving reparametrisations: hidden permutations and positive per-neuron rescalings
         (Wi_k, Wh[k,:] * c_k; Wh[:,k], Wo[:,k] / c_k). The step-2 impact w2_k = max_j |Wh[j,k]| is NOT invariant
         to them (it depends on the step-3 neurons' arbitrary scales), so the estimate can move although the
         accuracy cannot. Random c = e^U(-1,1), then a hill climb over log c in [-2, 2] maximising |error|.
 part 2  weight perturbations inside the box |dw| <= 0.1 (2 sigma) on all 32 weights, hill climb on |error|:
         (a) constrained to exact accuracy >= clean - 1 pp, (b) unconstrained.
usage: OMP_NUM_THREADS=1 python adversarial.py  -> adversarial.log / adversarial.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import exact as EX
import recipe as R
from mine import zoo_w, noisy_w


def est(m):
    return R.recipe(R.as_lists(m))


def rescale(m, logc, perm=None):
    Wi, Wh, Wo = (np.array(x, float) for x in m)
    c = np.exp(logc)
    Wi = Wi * c; Wh = (c[:, None] * Wh) / c[None, :]; Wo = Wo / c[None, :]
    if perm is not None:
        Wi = Wi[perm]; Wh = Wh[np.ix_(perm, perm)]; Wo = Wo[:, perm]
    return Wi, Wh, Wo


def climb(f, x0, lo, hi, steps, step, g):
    x = x0.copy(); fx = f(x)
    for _ in range(steps):
        y = np.clip(x + step * g.standard_normal(x.shape) * (g.random(x.shape) < 0.3), lo, hi)
        fy = f(y)
        if fy > fx:
            x, fx = y, fy
    return x, fx


def main():
    g = np.random.default_rng(84)        # drives the adversary only; the estimator makes no random calls
    t0 = time.time(); out = {}
    for zs in range(5):
        base = zoo_w(zs); tb = EX.exact_acc(base); eb = est(base)
        r = dict(clean_truth=tb, clean_err=eb - tb)
        # part 1: random reparametrisations
        models = [base] + noisy_w(base, 10, 0.05, 60)
        d_perm, d_resc, d_acc = 0.0, 0.0, 0.0
        for m in models:
            e0 = est(m)
            for _ in range(3):
                d_perm = max(d_perm, abs(est(rescale(m, np.zeros(4), g.permutation(4))) - e0))
                mr = rescale(m, g.uniform(-1, 1, 4))
                d_resc = max(d_resc, abs(est(mr) - e0))
            d_acc = max(d_acc, abs(EX.exact_acc(mr) - EX.exact_acc(m)))
        r.update(perm_maxdiff=d_perm, rescale_random_maxdiff=d_resc, rescale_truth_change=d_acc)
        # adversarial rescaling of the clean model and of 3 noisy copies
        worst = []
        for m in [base] + noisy_w(base, 3, 0.05, 61):
            t = EX.exact_acc(m)
            _, fx = climb(lambda lc: abs(est(rescale(m, lc)) - t), np.zeros(4), -2.0, 2.0, 80, 0.5, g)
            worst.append((abs(est(m) - t), fx))
        r["rescale_adv"] = worst
        # part 2: perturbation box
        flat = lambda m: np.concatenate([m[0], m[1].ravel(), m[2].ravel()])
        unflat = lambda v: (v[:4], v[4:20].reshape(4, 4), v[20:].reshape(3, 4))
        v0 = flat(base)

        def obj_c(dv):
            m = unflat(v0 + dv); t = EX.exact_acc(m)
            return abs(est(m) - t) if t >= tb - 0.01 else -1.0

        def obj_u(dv):
            m = unflat(v0 + dv); return abs(est(m) - EX.exact_acc(m))
        best_c = max(climb(obj_c, np.zeros(32), -0.1, 0.1, 100, 0.04, g)[1] for _ in range(3))
        res_u = [climb(obj_u, g.uniform(-0.1, 0.1, 32), -0.1, 0.1, 100, 0.04, g) for _ in range(3)]
        xu, best_u = max(res_u, key=lambda z: z[1])
        r.update(box_constrained_worst=best_c, box_unconstrained_worst=best_u,
                 box_unconstrained_acc=EX.exact_acc(unflat(v0 + xu)))
        out[zs] = r
        print(f"zoo {zs}: clean err {100*r['clean_err']:+.3f}pp | perm maxdiff {d_perm:.1e} | random rescale maxdiff "
              f"{100*d_resc:.3f}pp (truth change {d_acc:.1e}) | adversarial rescale |err| before->after (pp) "
              + ", ".join(f"{100*a:.2f}->{100*b:.2f}" for a, b in worst)
              + f" | box |dw|<=0.1: acc-preserving worst {100*best_c:.2f}pp, unconstrained worst {100*best_u:.2f}pp "
              f"(acc {r['box_unconstrained_acc']:.3f}) [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(os.path.join(HERE, "adversarial.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
