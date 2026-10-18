"""T232: exact prefix regions (width16_probe/prefix.py, unchanged), per-region exact masses at k >= 4 (estT.mass on
leaf cones, summed per gate word; diagnostic) and 1e6-sample 2nd-argmax accuracy at T = 10 (diagnostic) for the target
and its random / structure-destroyed controls (models defined in the README pre-registration).
usage: python rc.py model k [mass]  -> out/<model>_k{k}.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, "..", "width16_probe")
sys.path[:0] = [P, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
import estT
from m43 import Ops
from prefix import target, run


def model(name):
    Wi, Wh, Wo = target()
    if name == "target":
        return Wi, Wh, Wo
    kind, s = name[:-1], int(name[-1])
    if kind == "frob":
        r = np.random.default_rng(100 + s)
        f = lambda W: (lambda X: X * np.linalg.norm(W) / np.linalg.norm(X))(r.standard_normal(W.shape))
        return f(Wi), f(Wh), f(Wo)
    if kind == "row":
        r = np.random.default_rng(200 + s)
        def g(W):
            X = r.standard_normal(W.shape)
            return X * (np.linalg.norm(W, axis=-1, keepdims=True) / np.linalg.norm(X, axis=-1, keepdims=True))
        return np.abs(Wi) * np.sign(r.standard_normal(Wi.shape)), g(Wh), g(Wo)
    if name == "perm0":
        r = np.random.default_rng(300)
        return Wi, np.array([row[r.permutation(len(row))] for row in Wh]), Wo
    if name == "flip0":
        r = np.random.default_rng(301)
        return Wi, Wh * np.where(r.random(Wh.shape) < 0.5, -1.0, 1.0), Wo
    raise ValueError(name)


def acc(m, n=1_000_000, seed=7):
    Wi, Wh, Wo = m; x = np.random.default_rng(seed).standard_normal((n, 10)); h = np.zeros((n, len(Wi)))
    for t in range(10):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return float(((h @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]).mean()), float((h > 0).mean())


if __name__ == "__main__":
    name, k = sys.argv[1], int(sys.argv[2]); do_mass = len(sys.argv) > 3
    m = model(name); t0 = time.process_time()
    r, leaves = run(m, k, save=do_mass)
    if do_mass:
        w = {}
        for c, hist in leaves:
            w[hist] = w.get(hist, 0.0) + estT.mass(c, Ops())
        v = np.sort(np.array(list(w.values())))[::-1]; cs = np.cumsum(v) / v.sum()
        r.update(mass_sum_minus_1=float(v.sum() - 1), N={f"{d:.0e}": int(np.searchsorted(cs, 1 - d) + 1) for d in (1e-2, 1e-4, 1e-6)},
                 inv_simpson=float(v.sum() ** 2 / (v ** 2).sum()))
    if k == 2:
        r["acc_1e6"], r["active_frac_T10"] = acc(m)
    r.update(model=name, cpu_total=time.process_time() - t0)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(r, open(os.path.join(HERE, "out", f"{name}_k{k}.json"), "w"), indent=1)
    print(name, k, {q: r[q] for q in r if q not in ("model",)}, flush=True)
