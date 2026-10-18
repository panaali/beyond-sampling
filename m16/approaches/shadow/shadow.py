"""T290 (pre-registered in README): normalise, round (R mantissa / G grid, k = 1..8), merge identical units; per model the
unit count, paired-MC accuracy change and k_prefix=3 regions. usage (repo root):
python m16/approaches/shadow/shadow.py DEV2_WEIGHTS.npz -> shadow.json (resumable)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, "..", "width16_probe")
sys.path[:0] = [P, os.path.join(HERE, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
from prefix import run as prefix_run          # T216 exact prefix-region enumerator (unchanged)
T = 10


def normalise(m, iters=200):
    Wi, Wh, Wo = (W.copy() for W in m); Wo = Wo - Wo.mean(0, keepdims=True)
    for _ in range(iters):
        c = 1 / np.sqrt(Wi ** 2 + (Wh ** 2).sum(1)); Wi = c * Wi; Wh = c[:, None] * Wh / c[None, :]; Wo = Wo / c[None, :]
    res = float(np.abs(np.sqrt(Wi ** 2 + (Wh ** 2).sum(1)) - 1).max())
    return (Wi, Wh, Wo), res


def rnd(W, k, var):
    if var == "R":
        e = np.floor(np.log2(np.where(W == 0, 1, np.abs(W)))); s = 2.0 ** (e - k); return np.where(W == 0, 0, np.round(W / s) * s)
    s = 2.0 ** -k * np.abs(W).max(); return np.round(W / s) * s


def merge(m):
    Wi, Wh, Wo = m; H = len(Wi); blk = np.zeros(H, int)
    while True:
        nb = blk.max() + 1
        sig = [tuple(np.round(np.r_[Wi[j], [Wh[j, blk == b].sum() for b in range(nb)]], 12)) for j in range(H)]
        keys = {}; new = np.array([keys.setdefault((blk[j], sig[j]), len(keys)) for j in range(H)])
        if new.max() == blk.max():
            break
        blk = new
    nb = blk.max() + 1; rep = [int(np.where(blk == b)[0][0]) for b in range(nb)]
    Wi2 = Wi[rep]; Wh2 = np.array([[Wh[rep[a], blk == b].sum() for b in range(nb)] for a in range(nb)])
    Wo2 = np.array([Wo[:, blk == b].sum(1) for b in range(nb)]).T
    live = ~((Wi2 == 0) & (np.abs(Wh2).sum(1) == 0))                        # dead: all-zero signature
    # iterate deadness (a unit fed only by dead units is dead) and observability (feeds Wo or a live+observable unit)
    for _ in range(nb):
        live = (Wi2 != 0) | (np.abs(Wh2[:, live]).sum(1) > 0)
    obs = np.abs(Wo2).sum(0) > 0
    for _ in range(nb):
        obs = obs | (np.abs(Wh2[obs][:, :]).sum(0) > 0) & live
    keep = live & obs
    return (Wi2[keep], Wh2[np.ix_(keep, keep)], Wo2[:, keep]), int(keep.sum())


def correct(m, X, y):
    Wi, Wh, Wo = m; h = np.zeros((len(X), len(Wi)))
    for t in range(T):
        h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1) == y


_CACHE = {}


def acc_paired(m0, m1, n=1_000_000, ch=200_000):
    """same inputs (rng 290) for both; the original's correctness is cached per model (id of m0)."""
    rng = np.random.default_rng(290); a0 = a1 = 0; key = id(m0)
    for c in range(n // ch):
        X = rng.standard_normal((ch, T)); y = np.argsort(X, 1)[:, -2]
        if (key, c) not in _CACHE:
            _CACHE[(key, c)] = int(correct(m0, X, y).sum())
        a0 += _CACHE[(key, c)]; a1 += correct(m1, X, y).sum()
    return a0 / n, a1 / n


if __name__ == "__main__":
    z = np.load(sys.argv[1], allow_pickle=False); fn = os.path.join(HERE, "shadow.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    chk = np.random.default_rng(1).standard_normal((10_000, T))
    for i, name in enumerate(map(str, z["names"])):
        if name in out:
            continue
        t0 = time.process_time(); m = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
        _CACHE.clear(); mn, res = normalise(m); y = np.argsort(chk, 1)[:, -2]
        assert (correct(m, chk, y) == correct(mn, chk, y)).all()
        _, u0 = merge(mn); r0 = prefix_run(mn, 3)[0]["regions"]
        rec = dict(norm_residual=res, units_orig=u0, regions3_orig=r0, shadows={})
        for var in "RG":
            for k in range(1, 9):
                mr = tuple(rnd(W, k, var) for W in mn); ms, u = merge(mr)
                if u > 0:
                    L1 = np.maximum(0, 0); hh = np.zeros((len(chk), len(mr[0]))); hs = np.zeros((len(chk), u))
                    for t in range(T):
                        hh = np.maximum(hh @ mr[1].T + chk[:, t:t + 1] * mr[0][None], 0); hs = np.maximum(hs @ ms[1].T + chk[:, t:t + 1] * ms[0][None], 0)
                    assert np.allclose(hh @ mr[2].T, hs @ ms[2].T, atol=1e-8 * (1 + np.abs(hh @ mr[2].T).max()))
                    a0, a1 = acc_paired(mn, ms); r3 = prefix_run(ms, 3)[0]["regions"]
                else:
                    a0, a1 = acc_paired(mn, mn)[0], 0.1; r3 = 1
                rec["shadows"][f"{var}{k}"] = dict(units=u, dacc_pp=100 * (a1 - a0), acc_orig=a0, regions3=r3)
        rec["cpu"] = time.process_time() - t0; out[name] = rec
        print(name, "units0", u0, "reg0", r0, " ".join(f"{k}:u{v['units']},d{v['dacc_pp']:+.2f},r{v['regions3']}" for k, v in rec["shadows"].items()), "cpu %.0f" % rec["cpu"], flush=True)
        json.dump(out, open(fn, "w"), indent=1)
