"""T74 item 4: robustness / adversarial tests of tier B (and C). Truth = exact.py (validated by truth_mc.py).
  1. function-preserving reparametrisations (neuron permutation, positive per-neuron rescaling): accuracy is
     unchanged, so the estimate should be too.
  2. targeted scans: push a crossover tau slope across 1 (Wh[0,3] and Wh[2,0]), and break the mirror symmetry
     (noise on one mirror half only).
  3. worst case inside the noise box: random-restart hill climbing on |est_B - exact| over perturbations with
     every |dw| <= 0.10 (2 sigma of ARC's noise), with and without the constraint exact acc >= base - 1 pp.
  4. noisy copies (sigma 0.05) around zoo seeds 1-4.
usage: OMP_NUM_THREADS=1 python adversarial.py   (writes adversarial.json)"""
import json, os, time
import numpy as np
from common_review import base_model, zoo, noisy
import est_mech as EM
from m43 import Ops
from exact import exact_acc
from run_mech import TIERS

KB = dict(TIERS["B default (A2 + resolved crossover)"]); CB = KB.pop("complete")
KC = dict(TIERS["C resolved everywhere"]); CC = KC.pop("complete")


def estB(m):
    EM.COMPLETE = CB; return EM.est_mech(m, Ops(), **KB)


def estC(m):
    EM.COMPLETE = CC; return EM.est_mech(m, Ops(), **KC)


def pack(m):
    return np.concatenate([m[0], m[1].ravel(), m[2].ravel()])


def unpack(v):
    return v[:4].copy(), v[4:20].reshape(4, 4).copy(), v[20:].reshape(3, 4).copy()


out = {}; t0 = time.time()
b = base_model(); acc0 = exact_acc(b)
# ---------------------------------------------------------------- 1. reparametrisations
rng = np.random.default_rng(2024)
test = [b] + noisy(b, 20, 0.05, 50)
dmax = {"perm_B": 0, "scale_B": 0, "perm_exact": 0, "scale_exact": 0}
for m in test:
    Wi, Wh, Wo = m; eb = estB(m); ex = exact_acc(m)
    P = rng.permutation(4); mp = (Wi[P], Wh[np.ix_(P, P)], Wo[:, P])
    c = np.exp(rng.uniform(-1, 1, 4)); ms = (Wi * c, Wh * c[:, None] / c[None, :], Wo / c[None, :])
    dmax["perm_B"] = max(dmax["perm_B"], abs(estB(mp) - eb)); dmax["scale_B"] = max(dmax["scale_B"], abs(estB(ms) - eb))
    dmax["perm_exact"] = max(dmax["perm_exact"], abs(exact_acc(mp) - ex)); dmax["scale_exact"] = max(dmax["scale_exact"], abs(exact_acc(ms) - ex))
out["reparam_max_abs_change"] = dmax
print("1. reparametrisation, max |change| over 21 models:", {k: f"{v:.2e}" for k, v in dmax.items()}, flush=True)


# ---------------------------------------------------------------- 2a. crossover slope scans
def slope(m, k, sigma=1.0, strip=0):
    """beta_k = d tau_k / d x1 in the given step-2 strip of chart sigma (tau_k = x2 threshold of neuron k)."""
    Wi, Wh, Wo = m; E = np.eye(3)
    g1 = ((sigma * Wi) > 0).astype(float)
    A2 = Wh @ (g1[:, None] * np.outer(Wi, E[0])) + np.outer(Wi, E[1])
    ts = np.sort(-A2[:, 0] / Wi); bnd = [-np.inf] + list(ts) + [np.inf]
    lo, hi = bnd[strip], bnd[strip + 1]
    r = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else (hi - 1 if np.isfinite(hi) else lo + 1)
    g2 = (sigma * (A2[:, 0] + Wi * r) > 0).astype(float)
    A3 = Wh @ (g2[:, None] * A2) + np.outer(Wi, E[2])
    return -A3[k, 1] / A3[k, 2]


scans = {}
for (ii, jj, kk, strip) in ((0, 3, 0, 0), (2, 3, 2, 0), (0, 3, 0, 4), (1, 1, 1, 4)):  # NB the 3rd scan perturbs the same weight as the 1st; only the reported slope differs
    rowsS = []
    for d in np.linspace(-0.4, 0.4, 17):
        Wi, Wh, Wo = [x.copy() for x in b]; Wh[ii, jj] += d; m = (Wi, Wh, Wo)
        ex = exact_acc(m); eb = estB(m); ec = estC(m)
        rowsS.append(dict(d=float(d), slope=float(slope(m, kk, 1.0, strip)), exact=ex, errB=eb - ex, errC=ec - ex))
    scans[f"Wh[{ii},{jj}] -> beta_{kk} (chart +, strip {strip})"] = rowsS
    print(f"2a. scan Wh[{ii},{jj}] (beta_{kk}, strip {strip}):")
    for r_ in rowsS:
        print(f"    d {r_['d']:+.2f} beta {r_['slope']:.3f} exact {r_['exact']:.4f} errB {r_['errB']:+.4f} errC {r_['errC']:+.4f}")
out["slope_scans"] = scans

# ---------------------------------------------------------------- 2b. one-sided (mirror-breaking) noise
res = {}
for sig in (0.05, 0.1, 0.2):
    r2 = np.random.default_rng(int(sig * 1000)); eB = []; eC = []; accs = []
    for _ in range(128):
        Wi, Wh, Wo = [x.copy() for x in b]
        for k in (0, 1):                                  # only the u half (neurons 0, 1): rows, cols, readout
            Wi[k] += sig * r2.standard_normal(); Wh[k, :] += sig * r2.standard_normal(4)
            Wh[:, k] += sig * r2.standard_normal(4); Wo[:, k] += sig * r2.standard_normal(3)
        m = (Wi, Wh, Wo); ex = exact_acc(m); accs.append(ex); eB.append(estB(m) - ex); eC.append(estC(m) - ex)
    eB, eC, accs = map(np.array, (eB, eC, accs)); pq = np.mean(accs * (1 - accs))
    res[sig] = dict(mean_acc=float(accs.mean()), rmsB=float(np.sqrt(np.mean(eB ** 2))), maxB=float(np.abs(eB).max()),
                    rmsC=float(np.sqrt(np.mean(eC ** 2))), ratioB_at_14k=float(np.mean(eB ** 2) / (pq / (14000 / 72))))
    print(f"2b. one-sided noise sigma {sig}: mean acc {accs.mean():.4f} B rms {res[sig]['rmsB']:.4f} max {res[sig]['maxB']:.4f} "
          f"C rms {res[sig]['rmsC']:.4f}; B ratio (at 14k ops) {res[sig]['ratioB_at_14k']:.3f}", flush=True)
out["one_sided_noise"] = res


# ---------------------------------------------------------------- 3. worst case in the box |dw| <= 0.10
def climb(seed, box=0.10, keep_acc=False, iters=150):
    r3 = np.random.default_rng(seed); v0 = pack(b)
    # constrained climbs start at the base model (a random box point almost never keeps accuracy within 1 pp)
    dv = np.zeros(32) if keep_acc else np.clip(0.05 * r3.standard_normal(32), -box, box)
    def score(dv):
        m = unpack(v0 + dv); ex = exact_acc(m)
        if keep_acc and ex < acc0 - 0.01:
            return -1.0, ex, 0.0
        e = estB(m) - ex; return abs(e), ex, e
    best = score(dv)
    for it in range(iters):
        cand = np.clip(dv + 0.02 * r3.standard_normal(32) * (r3.random(32) < 0.25), -box, box)
        sc = score(cand)
        if sc[0] > best[0]:
            dv, best = cand, sc
    return dict(seed=seed, abs_err=best[0], exact=best[1], err=best[2], dv=dv.tolist())


wc = {}
for keep in (False, True):
    rs = [climb(100 + s, keep_acc=keep) for s in range(6)]
    j = int(np.argmax([r_["abs_err"] for r_ in rs])); wc["keep_acc" if keep else "free"] = rs
    print(f"3. worst case |dw|<=0.10 {'acc >= base-1pp' if keep else 'unconstrained'}: per restart |err| "
          f"{[round(float(r_['abs_err']), 4) for r_ in rs]}; worst {rs[j]['err']:+.4f} at exact acc {rs[j]['exact']:.4f}", flush=True)
    mw = unpack(pack(b) + np.array(rs[j]["dv"]))
    print(f"   worst model: tier C err {estC(mw) - rs[j]['exact']:+.4f}; crossover slopes chart+ strip0 "
          f"{[round(float(slope(mw, k, 1.0, 0)), 3) for k in range(4)]}", flush=True)
out["worst_case"] = wc

# ---------------------------------------------------------------- 4. noisy copies of zoo seeds 1-4
zres = {}
for s in range(1, 5):
    ms = noisy(zoo(s), 64, 0.05, 60 + s); tr = np.array([exact_acc(m) for m in ms])
    eB = np.array([estB(m) for m in ms]) - tr; eC = np.array([estC(m) for m in ms]) - tr
    pq = np.mean(tr * (1 - tr))
    zres[s] = dict(mean_acc=float(tr.mean()), rmsB=float(np.sqrt(np.mean(eB ** 2))), biasB=float(eB.mean()),
                   rmsC=float(np.sqrt(np.mean(eC ** 2))), biasC=float(eC.mean()),
                   ratioB=float(np.mean(eB ** 2) / (pq / (14000 / 72))), ratioC=float(np.mean(eC ** 2) / (pq / (23000 / 72))))
    print(f"4. zoo seed {s} + noise 0.05 (64 copies): acc {tr.mean():.4f}  B rms {zres[s]['rmsB']:.4f} bias {zres[s]['biasB']:+.4f} "
          f"ratio~{zres[s]['ratioB']:.2f} | C rms {zres[s]['rmsC']:.4f} bias {zres[s]['biasC']:+.4f} ratio~{zres[s]['ratioC']:.2f}", flush=True)
out["zoo_noisy"] = zres
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "adversarial.json"), "w"), indent=1, default=float)
print(f"done {time.time()-t0:.0f}s")
