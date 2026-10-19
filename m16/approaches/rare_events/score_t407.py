"""T407 scoring: graded-g12 estimates (results/t407_est.json) vs T401's conditional MC checks for orig (results/t401_mc.json);
rate only within 2x; deterministic flag vs sampled tie share; F vs both bars. -> results/t407_score.txt"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
e = json.load(open(os.path.join(HERE, "results", "t407_est.json"))); mc = json.load(open(os.path.join(HERE, "results", "t401_mc.json")))
k3 = json.load(open(os.path.join(HERE, "results", "t401_est.json")))["rows"]["orig"]
C = e["C"]; L = []; fl_d, fl_s = [], []
L.append(f"{'t':>4} {'P(cond)':>9} {'rating r(t)':>11} {'check':>9} {'est/check':>9} {'K3 est/check':>12} {'flag det':>8} {'flag MC':>7} {'label':>18} {'F':>8}")
for t in ("0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5"):
    m = mc[f"orig:E1:{t}"]; P = m["P_cond"]; est = e["est"][t] / P; fd = e["flag_share_g12_lt_0p1"][t]; fs = m["tie_share"]
    ok = m["hits"] >= 30
    if ok:
        q = m["rate"]; ratio = est / q; rk = k3["E1"][t] / P / q
        label = "rated" if 0.5 <= ratio <= 2 else "unrated (off >2x)"
        p = q * P; F = (e["est"][t] - p) ** 2 * max(C, 1000) / (p * (1 - p))
        fl_d.append(fd); fl_s.append(fs)
        L.append(f"{t:>4} {P:9.3g} {est:11.3g} {q:9.3g} {ratio:9.2f} {rk:12.3g} {fd:8.2f} {fs:7.2f} {label:>18} {F:8.3g}")
    else:
        L.append(f"{t:>4} {P:9.3g} {est:11.3g} {'<30 hits':>9} {'-':>9} {'-':>12} {fd:8.2f} {'-':>7} {'beyond test':>18} {'-':>8}")
fd, fs = np.array(fl_d), np.array(fl_s)
L.append(f"\nC = {C:.4g} FPE (wrapper equality max rel diff {e['max_rel_diff']:.1e}). Flags at validated t: corr {np.corrcoef(fd, fs)[0, 1]:.2f}, "
         f">=50% call agrees on {int(((fd >= .5) == (fs >= .5)).sum())}/{len(fd)}; mean |det - MC| {np.abs(fd - fs).mean():.2f}")
open(os.path.join(HERE, "results", "t407_score.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
