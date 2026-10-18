"""T337: T317f's noise breakdown curve (README, T337 pre-registration).
usage: python noise.py truths   -> results_noise.json (truths + error decomposition + T317f estimates, all models)
       python noise.py switch   -> results_noise_switch.json (S1-S3 at the first level with F > 10)"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
import r0  # noqa: E402
import estimator_f  # noqa: E402

LEVELS = (0.0, 0.0005, 0.001, 0.0025, 0.005)
NCOPY = 4
TMAX = 0.5
OUT = os.path.join(HERE, "results_noise.json")


def bases():
    from alg_zoo import zoo_2nd_argmax
    f = lambda w: w.detach().numpy().astype(np.float64)
    out = {"target": r0.target()}
    for sd in range(5):
        m = zoo_2nd_argmax(16, 10, seed=sd)
        out[f"zoo{sd}"] = (f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight))
    return out


def copies(base, bi, li, sigma):
    if sigma == 0:
        return [base]
    rng = np.random.default_rng(337000 + 100 * bi + 10 * li); out = []
    for _ in range(NCOPY):
        out.append(tuple(W + sigma * np.sqrt((W ** 2).mean()) * rng.standard_normal(W.shape) for W in base))
    return out


def truth(model, n=10_000_000, chunk=250_000, seed=1, dtype=np.float32):
    """Accuracy plus the misclassified mass split by window (g12 or g23 < TMAX) / interior, predicted rank, v2 sign."""
    Wi, Wh, Wo = (np.asarray(w, dtype) for w in model)
    rng = np.random.default_rng(seed); ok = 0
    dec = dict(win=0, interior=0, pred_top1=0, pred_top3=0, pred_other=0, blind=0, nonblind=0,
               interior_top1=0, interior_top3=0, interior_other=0)
    for s in range(0, n, chunk):
        x = rng.standard_normal((min(chunk, n - s), 10), dtype=dtype); h = np.zeros((len(x), 16), dtype)
        for t in range(10):
            h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi, 0)
        pred = (h @ Wo.T).argmax(1)
        order = np.argsort(-x, 1); xs = np.take_along_axis(x, order, 1)
        y = order[:, 1]; bad = pred != y; ok += int((~bad).sum())
        g12 = xs[:, 0] - xs[:, 1]; g23 = xs[:, 1] - xs[:, 2]
        win = (g12 < TMAX) | (g23 < TMAX)
        rank = np.argmax(order == pred[:, None], 1)                   # predicted position's rank (0 = top-1)
        b = bad
        dec["win"] += int((b & win).sum()); dec["interior"] += int((b & ~win).sum())
        dec["pred_top1"] += int((b & (rank == 0)).sum()); dec["pred_top3"] += int((b & (rank == 2)).sum())
        dec["pred_other"] += int((b & (rank >= 3)).sum())
        dec["interior_top1"] += int((b & ~win & (rank == 0)).sum()); dec["interior_top3"] += int((b & ~win & (rank == 2)).sum())
        dec["interior_other"] += int((b & ~win & (rank >= 3)).sum())
        dec["blind"] += int((b & (xs[:, 1] <= 0)).sum()); dec["nonblind"] += int((b & (xs[:, 1] > 0)).sum())
    return ok / n, {k: v / n for k, v in dec.items()}


def run_truths():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    # float32 vs float64 check
    if "check32" not in res:
        B = bases(); chk = {}
        for nm in ("target", "zoo0"):
            m = copies(B[nm], list(B).index(nm), 4, 0.005)[0]
            a32, _ = truth(m, n=1_000_000, dtype=np.float32); a64, _ = truth(m, n=1_000_000, dtype=np.float64)
            chk[nm] = [a32, a64]
        res["check32"] = chk; json.dump(res, open(OUT, "w"), indent=1); print("check32", chk, flush=True)
    B = bases(); t0 = time.time()
    for bi, (bn, base) in enumerate(B.items()):
        for li, sg in enumerate(LEVELS):
            for ci, m in enumerate(copies(base, bi, li, sg)):
                key = f"{bn}|{sg}|{ci}"
                if key in res:
                    continue
                acc, dec = truth(m); e, f = estimator_f.est(m)
                res[key] = dict(base=bn, sigma=sg, copy=ci, truth=acc, dec=dec, est=e, C=f / 5948)
                json.dump(res, open(OUT, "w"), indent=1)
                print(key, round(acc, 5), round(e, 5), round(100 * (e - acc), 3), round(f / 5948), round(time.time() - t0), flush=True)


def summarise():
    res = json.load(open(OUT)); out = {}
    for sg in LEVELS:
        R = [v for k, v in res.items() if k != "check32" and v["sigma"] == sg]
        e = np.array([r["est"] - r["truth"] for r in R]); p = np.array([r["truth"] for r in R]); C = np.mean([r["C"] for r in R])
        err = np.array([1 - r["truth"] for r in R]); D = {k: np.mean([r["dec"][k] for r in R]) for k in R[0]["dec"]}
        out[str(sg)] = dict(n=len(R), truth_mean=float(p.mean()), truth_min=float(p.min()), bias_pp=100 * float(e.mean()),
                            rms_pp=100 * float(np.sqrt((e ** 2).mean())), C=float(C), F=float((e ** 2).mean() * C / np.mean(p * (1 - p))),
                            err_pp=100 * float(err.mean()), **{f"{k}_pp": 100 * float(v) for k, v in D.items()})
    return out


def est_switch(model, c1=True, tmax=0.5, blind_walk=False):
    """T317f with one declared approximation switched off: c1=False (S1, = T317e), tmax=6 (S2), blind_walk (S3).
    Defaults reproduce estimator_f.est's estimate bitwise."""
    import cont_e as E
    import screen_c as S
    model = tuple(np.asarray(w, np.float64) for w in model); k = E.TRI[:, 1]; tot = np.zeros(E.NT)
    for tb in E.TABLES:
        nodes = tb["nodes"]; I = {}
        lines = [("23", tb["line23"])]
        if "line12" in tb:
            lines.append(("12", tb["line12"]))
        elif blind_walk:
            v2s, g12s, g23s = tb["point"]; v3s = v2s - g23s
            lines.append(("12", E.inputs(v2s, v2s, v3s, tb["v4"], tb["m4"], 1.0, 0.0, 0.0, 0.0)))
        for kind, (x, dx) in lines:
            p, _, _ = S.walk_screen(model, x, dx, k, tmax, c1, mode="C1")
            R, A, B, _ = E.merge(p, E.NT, tmax)
            I[kind], _ = E.profile(R, A, B, E.NT, nodes, kind, tmax, tb["phi0"])
        I12 = I["12"] if "12" in I else np.repeat(tb["blind12"][None], E.NT, 0)
        tot += (I12 * I["23"] * tb["dens"][None]) @ tb["wts"]
    return float(tot.mean())


def run_switch(level):
    res = json.load(open(OUT)); B = bases(); out = {}
    for bn in ("target", "zoo0"):
        bi = list(B).index(bn); li = LEVELS.index(level); m = copies(B[bn], bi, li, level)[0]
        r = res[f"{bn}|{level}|0"]
        base = est_switch(m); assert base == r["est"], (base, r["est"])
        out[bn] = dict(truth=r["truth"], T317f=base, S1_c1_off=est_switch(m, c1=False), S2_tmax6=est_switch(m, tmax=6.0),
                       S3_blind_walk=est_switch(m, blind_walk=True), dec=r["dec"])
        out[bn]["err_pp"] = {k: 100 * (v - r["truth"]) for k, v in out[bn].items() if k not in ("truth", "dec", "err_pp")}
        print(bn, out[bn]["err_pp"], flush=True)
    json.dump(dict(level=level, rows=out), open(os.path.join(HERE, "results_noise_switch.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "switch":
        run_switch(float(sys.argv[2]))
    if sys.argv[1] == "truths":
        run_truths(); s = summarise(); print(json.dumps(s, indent=1))
        json.dump(s, open(os.path.join(HERE, "results_noise_summary.json"), "w"), indent=1)
