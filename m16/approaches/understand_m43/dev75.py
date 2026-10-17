"""T75 development harness (dev noise seeds 0-1 only; fresh seeds 6-9 are reserved for the frozen evaluation).
usage: python dev75.py [n_per_seed] [noise_seed]   prints per zoo seed: clean error, noisy RMSE/bias/ops/ratio for est"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
import numpy as np
from m43 import Ops, OPS_PER_SAMPLE
from exact import exact_acc
from zoo43 import zoo_model, noisy_copies

CACHE = os.path.join(HERE, "dev75_truth.json")


def truths(n, ns):
    c = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    out = {}
    for s in range(5):
        key = f"{s}_{ns}_{n}"
        if key not in c:
            c[key] = [exact_acc(m) for m in noisy_copies(zoo_model(s), n, 0.05, ns)]
        out[s] = np.array(c[key])
    json.dump(c, open(CACHE, "w"))
    return out


def evaluate(est, n=64, ns=0, verbose=True):
    T = truths(n, ns); res = {}
    for s in range(5):
        base = zoo_model(s); ob = Ops(); eb = est(base, ob) - exact_acc(base)
        ms = noisy_copies(base, n, 0.05, ns); e = []; ops = []
        for m in ms:
            o = Ops(); e.append(est(m, o)); ops.append(o.n)
        e = np.array(e) - T[s]; om = np.mean(ops); pq = np.mean(T[s] * (1 - T[s]))
        ratio = np.mean(e ** 2) / (pq / (om / OPS_PER_SAMPLE))
        pb = exact_acc(base); rb = eb ** 2 / (pb * (1 - pb) / (ob.n / OPS_PER_SAMPLE))
        res[s] = dict(clean_err=eb, clean_ops=ob.n, clean_ratio=rb, rms=np.sqrt(np.mean(e ** 2)), bias=e.mean(), ops=om, ratio=ratio,
                      maxe=np.abs(e).max(), mean_truth=T[s].mean())
        if verbose:
            print(f" seed {s}: clean err {100*eb:+.2f}% ({ob.n} ops, ratio {rb:.2f}) | noisy acc {T[s].mean():.3f}: rms {100*res[s]['rms']:.2f}% "
                  f"bias {100*e.mean():+.2f}% max {100*np.abs(e).max():.1f}% ops {om:.0f} ratio {ratio:.3f}")
    return res


if __name__ == "__main__":
    import est_mech as EM
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    kw = dict(strips="all", resolve=("c01", "c02", "c12"), pair_rule="arc", arc_fams=("c01", "c02", "c12"))
    print("tier C (T63):"); evaluate(lambda m, o: EM.est_mech(m, o, **kw), n, ns)
