"""T115: margin-aware recipes RECIPE85B and RECIPE112 (pinned est75_t112.py @ 9243921) on the training axis.
  panel 'p20': the T101 panels (noise seeds 20-23 x 256 per zoo seed, same copy order as run_rung.py); exact truths
               are reused from rung_k{k}.json (they were computed by exact.py there) and re-checked on 8 copies.
  panel 'p100': FRESH noise seeds 100-101 x 256 per zoo seed (only for k in FRESH_K); exact truth computed here; the
               frozen T75 RECIPE (pinned est75_e37.py) is scored on it too.
Ops: est75's strict charge. MSE ratio = MSE / (mean p(1-p) / (ops/72)); bootstrap CI over copies.
usage: OMP_NUM_THREADS=1 python run_margin.py k  -> margin_k{k}.json"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

import lt
from m43 import Ops
from run_rung import ratio_stats
from zoo43 import noisy_copies

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("est75_t112", os.path.join(HERE, "est75_t112.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
RECIPES = {"r85b": P.RECIPE85B, "r112": P.RECIPE112}
FRESH_K = (24, 30)
FRESH_SEEDS = (100, 101)


def run(m, nm):
    P.STATS.update(cut=0, resolved=0, cells=0)
    o = Ops(); r = P.est75(m, o, **RECIPES[nm])
    return float(r), o.n, P.STATS["cells"]


def score(models, tr, with_t75=False):
    out = {}
    names = list(RECIPES) + (["t75"] if with_t75 else [])
    for nm in names:
        est = np.empty(len(models)); ops = np.empty(len(models)); cells = np.empty(len(models))
        for i, m in enumerate(models):
            if nm == "t75":
                est[i], ops[i], st = lt.t75(m); cells[i] = st["cells"]
            else:
                est[i], ops[i], cells[i] = run(m, nm)
        d = ratio_stats(est - tr, tr, ops); d["cells"] = float(cells.mean()); d["errs"] = (est - tr).tolist()
        out[nm] = d
    return out


def main():
    k = int(sys.argv[1]); t0 = time.time()
    rung = json.load(open(os.path.join(HERE, f"rung_k{k}.json")))
    res = {"k": k, "seeds": {}}
    for s in range(5):
        R = rung["seeds"][str(s)]
        base = lt.zoo(k, s); pan = lt.panel(base)
        tr = np.array(R["panel"]["truths"])
        chk = max(abs(lt.truth(pan[i])[0] - tr[i]) for i in range(0, len(pan), 128))      # 8 spot checks
        assert chk < 1e-12, chk
        d = {"truth_spotcheck_maxdiff": float(chk), "regions_clean": R["clean"]["regions"], "regions_panel": R["panel"]["regions_mean"],
             "exact_ops": R["panel"]["exact_ops"], "t75_p20": {kk: R["panel"]["t75"][kk] for kk in ("ratio", "ci", "ops", "rms", "max_abs", "frac_copies_ratio_ge1")}}
        ctr = R["clean"]["truth"]; d["clean"] = {}
        for nm in RECIPES:
            e, o, c = run(base, nm); pq = ctr * (1 - ctr)
            d["clean"][nm] = dict(err=e - ctr, ops=o, cells=c, ratio=(e - ctr) ** 2 / (pq / (o / 72)))
        d["p20"] = score(pan, tr)
        if k in FRESH_K:
            fr = [m for ns in FRESH_SEEDS for m in noisy_copies(base, 256, lt.SIGMA, ns)]
            ftr = np.empty(len(fr)); fops = np.empty(len(fr))
            for i, m in enumerate(fr):
                ftr[i], fops[i] = lt.truth(m)
            d["p100"] = score(fr, ftr, with_t75=True); d["p100_exact_ops"] = float(fops.mean()); d["p100_mean_truth"] = float(ftr.mean())
        res["seeds"][s] = d
        line = f"k={k} seed {s}: regions {d['regions_clean']} exact {d['exact_ops']:.0f} | T75 {d['t75_p20']['ratio']:.4f}"
        for nm in RECIPES:
            q = d["p20"][nm]
            line += f" | {nm} {q['ratio']:.4f} [{q['ci'][0]:.4f},{q['ci'][1]:.4f}] ops {q['ops']:.0f} cells {q['cells']:.0f} clean err {100*d['clean'][nm]['err']:+.3f}pp"
        if "p100" in d:
            line += " || fresh 100-101: " + ", ".join(f"{nm} {d['p100'][nm]['ratio']:.4f}" for nm in ("t75", "r85b", "r112"))
        print(line + f" ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(os.path.join(HERE, f"margin_k{k}.json"), "w"), default=float)
    print(f"done k={k} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
