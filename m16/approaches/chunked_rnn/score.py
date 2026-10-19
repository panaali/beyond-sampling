"""T392 step 2 on one config k{K}h{H} (pre-registered in README): panel, 1e8 truths, gate patterns, random
great-circle F (+ direct checks at C = 4k, 16k FPE), fixed scrambled-Halton F at C = 4k, 16k.

usage: python score.py K H N_CIRCLES      -> results/score_k{K}h{H}.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cwalks  # noqa: E402
from cwalks import fast1, walks  # noqa: E402

T, CHUNK, SIGMA, CS = 10, 250_000, 0.001, (4000, 16000)


def panel(k, H):
    rng, out = np.random.default_rng(392), {}
    for s in range(3):
        m = cwalks.load(f"k{k}h{H}_s{s}")
        out[f"k{k}h{H}_s{s}"] = m
        ws = (m.Wih, m.Whh, m.Wout)
        for i in range(3):
            out[f"k{k}h{H}_s{s}_c{i}"] = cwalks.ChunkRNN(k, *[w + SIGMA * np.sqrt((w ** 2).mean()) * rng.standard_normal(w.shape) for w in ws])
    return out


def accuracy(m, n, seed):
    rng, ok = np.random.default_rng(seed), 0
    for _ in range(n // CHUNK):
        x = rng.standard_normal((CHUNK, T))
        ok += int((m.forward(x).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
    return ok / n


def patterns(m, n=10_000_000, seed=392002):
    """Distinct cumulative gate patterns after each step (64-bit hashes) in n inputs; share of last-step patterns seen once."""
    mult = np.random.default_rng(64).integers(1, 2 ** 62, size=m.S * m.H, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)
    rng, keys = np.random.default_rng(seed), [[] for _ in range(m.S)]
    for _ in range(n // CHUNK):
        g = m.gates(rng.standard_normal((CHUNK, T))).astype(np.uint64)
        acc = np.zeros(CHUNK, dtype=np.uint64)
        for t in range(m.S):
            sl = slice(t * m.H, (t + 1) * m.H)
            acc = acc * np.uint64(1000003) + (g[:, sl] * mult[sl]).sum(1)
            keys[t].append(acc.copy())
    out = []
    for t in range(m.S):
        u, c = np.unique(np.concatenate(keys[t]), return_counts=True)
        out.append({"distinct": int(len(u)), "seen_once": float((c == 1).sum() / len(u))})
    return out


def boot(num, den, n_boot=10000, seed=0):
    idx = np.random.default_rng(seed).integers(len(num), size=(n_boot, len(num)))
    b = num[idx].mean(1) / den[idx].mean(1)
    return [float(num.mean() / den.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def circle(m, u, v):
    r1, r2 = walks.walk(m, u, v, *walks.CAUCHY), walks.walk(m, -u, -v, *walks.CAUCHY)
    return (0.5 * (r1[0] + r2[0]),) + tuple(a + b for a, b in zip(r1[1:], r2[1:]))


def main(k, H, n_circ):
    path = f"{HERE}/results/score_k{k}h{H}.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    save = lambda: json.dump(res, open(path, "w"), indent=1)  # noqa: E731
    models = panel(k, H)
    names = list(models)
    res["fpe"] = cwalks.fpe(k, H)
    res.setdefault("truth", {}); res.setdefault("patterns", {}); res.setdefault("random", {})
    for i, (name, m) in enumerate(models.items()):
        t0 = time.time()
        if name not in res["truth"]:
            res["truth"][name] = accuracy(m, 100_000_000, 392003)
        if "_c" not in name and name not in res["patterns"]:
            res["patterns"][name] = patterns(m)
        if name not in res["random"]:
            rng = np.random.default_rng([392, k, i])
            v = np.array([circle(m, rng.standard_normal(T), rng.standard_normal(T)) for _ in range(n_circ)])
            p = res["truth"][name]
            rec = {"n": n_circ, "mean": float(v[:, 0].mean()), "var": float(v[:, 0].var(ddof=1)),
                   "cost_fpe": float(v[:, 1].mean() / m.fwd_flops), "pieces": float(v[:, 2].mean()),
                   "cost_incr_fpe": float(v[:, 4].mean() / m.fwd_flops),
                   "cost_floor_fpe": float((2 * m.fwd_flops + v[:, 2] * 4 * T + v[:, 3]).mean() / m.fwd_flops),
                   "z_vs_truth": float((v[:, 0].mean() - p) / np.sqrt(v[:, 0].var(ddof=1) / n_circ))}
            for C in CS:   # direct checks: consecutive groups of floor(C / c) circles, implemented and incremental cost
                for key, c in (("direct", rec["cost_fpe"]), ("direct_incr", rec["cost_incr_fpe"])):
                    g = max(1, int(C // c)); ng = n_circ // g
                    est = v[:ng * g, 0].reshape(ng, g).mean(1)
                    rec[f"{key}_{C}"] = {"group": g, "n_est": ng, "mse": float(np.mean((est - p) ** 2)),
                                         "C_eff": max(1000.0, g * c)}
            if k == 10:   # fast1 (implemented incremental walk), same circles
                rng = np.random.default_rng([392, k, i])
                f = np.array([fast1.circle(m.Wih, m.Wout, rng.standard_normal(T), rng.standard_normal(T)) for _ in range(n_circ)])
                rec["fast1"] = {"mean": float(f[:, 0].mean()), "var": float(f[:, 0].var(ddof=1)),
                                "cost_fpe": float(f[:, 1].mean() / m.fwd_flops)}
            res["random"][name] = rec
        save()
        print(name, f"{time.time() - t0:.0f}s", res["truth"][name], json.dumps(res["random"][name])[:300], flush=True)
    pq = np.array([res["truth"][n] * (1 - res["truth"][n]) for n in names])
    R = res["random"]
    summ = {"acc_mean": float(np.mean([res["truth"][n] for n in names])),
            "pieces": float(np.mean([R[n]["pieces"] for n in names])),
            "cost_fpe": float(np.mean([R[n]["cost_fpe"] for n in names]))}
    for key, ck in (("F", "cost_fpe"), ("F_incr", "cost_incr_fpe"), ("F_floor", "cost_floor_fpe")):
        summ[key] = boot(np.array([R[n]["var"] * R[n][ck] for n in names]), pq)
    for C in CS:
        for key in ("direct", "direct_incr"):
            summ[f"F_{key}_{C}"] = boot(np.array([R[n][f"{key}_{C}"]["mse"] * R[n][f"{key}_{C}"]["C_eff"] for n in names]), pq)
    if k == 10:
        summ["F_fast1"] = boot(np.array([R[n]["fast1"]["var"] * R[n]["fast1"]["cost_fpe"] for n in names]), pq)
    res["summary_random"] = summ
    save()
    print("random", json.dumps(summ), flush=True)
    # fixed scrambled-Halton sets
    res["halton"] = {}
    walkers = [("generic", summ["cost_fpe"])] + ([("fast1", float(np.mean([R[n]["fast1"]["cost_fpe"] for n in names])))] if k == 10 else [])
    for wname, cbar in walkers:
        for C in CS:
            n = max(1, round(C / cbar))
            for s in range(1, 6):
                Q = qmc.Halton(d=20, scramble=True, seed=s).random(n)
                U, V = ndtri(Q[:, :10]), ndtri(Q[:, 10:])
                e2, ceff = [], []
                for name, m in models.items():
                    if wname == "generic":
                        r = [circle(m, u, v)[:2] for u, v in zip(U, V)]
                    else:
                        r = [fast1.circle(m.Wih, m.Wout, u, v) for u, v in zip(U, V)]
                    e2.append((np.mean([a for a, _ in r]) - res["truth"][name]) ** 2)
                    ceff.append(max(1000.0, sum(f for _, f in r) / m.fwd_flops))
                e2, ceff = np.array(e2), np.array(ceff)
                F = boot(e2 * ceff, pq)
                res["halton"][f"{wname}_C{C}_H{s}"] = {"n": n, "F": F, "rms_err": float(np.sqrt(e2.mean())),
                                                       "cost_fpe": float(ceff.mean())}
                save()
                print(wname, C, s, n, F, flush=True)
            Fs = [res["halton"][f"{wname}_C{C}_H{s}"]["F"][0] for s in range(1, 6)]
            res["halton"][f"{wname}_C{C}_family"] = {"F_sets": Fs, "F_mean": float(np.mean(Fs)), "all_below_1": bool(max(Fs) < 1)}
            save()
            print("family", wname, C, Fs, flush=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
