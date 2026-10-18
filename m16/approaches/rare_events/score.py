"""T340 scoring: est / truth ratio, F = (est - p)^2 max(C, 1k) / (p (1 - p)), and p C (events plain sampling sees at C).
Truths: results/truth.json (E1), results/truth2.json (E2), results/truth_gpu.json if present (preferred where n larger).
-> results/score.json and a table"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda f: json.load(open(os.path.join(HERE, "results", f))) if os.path.exists(os.path.join(HERE, "results", f)) else {}
est = R("est.json"); t1 = R("truth.json"); t2 = R("truth2.json"); tg = R("truth_gpu.json")


def truth(model, ev, t):
    best = None
    for src, key in ((t1, "E1"), (t2, "E2"), (tg.get("E1", {}), "E1"), (tg.get("E2", {}), "E2")):
        if key != ev or model not in src:
            continue
        d = src[model]; c = d["counts"].get(str(t))
        if c is None:
            continue
        if best is None or d["n"] > best[1]:
            best = (c, d["n"])
    return best


rows = []; print(f"{'event':10s} {'model':6s} {'var':3s} {'truth (count/n)':>24s} {'est':>10s} {'ratio':>7s} {'C FPE':>8s} {'F':>10s} {'p*C':>8s}")
for ev, ts, key in (("E1", ("0.0", "1.0", "1.25", "1.5"), "est"), ("E2", ("0.5", "0.75"), "est2")):
    for t in ts:
        for model in ("orig", "seed0", "seed3"):
            tr = truth(model, ev, t)
            if tr is None:
                continue
            p = tr[0] / tr[1]; se = np.sqrt(tr[0]) / tr[1]
            for v in ("A", "B"):
                e = est[f"{model}:{v}"][key][t]; C = est[f"{model}:{v}"]["C"]
                F = (e - p) ** 2 * max(C, 1000) / (p * (1 - p))
                rows.append(dict(event=f"{ev}({t})", model=model, variant=v, truth=p, truth_se=se, count=tr[0], n=tr[1],
                                 est=e, ratio=e / p, C=C, F=F, pC=p * C))
                print(f"{ev}({t}):".ljust(10) + f" {model:6s} {v:3s} {p:10.3e} ({tr[0]:>7d}/{tr[1]:.0e}) {e:10.3e} {e / p:7.3f} {C:8.0f} {F:10.3g} {p * C:8.3g}")
json.dump(rows, open(os.path.join(HERE, "results", "score.json"), "w"), indent=1)
for v in ("A", "B"):
    for ev in ("E1", "E2"):
        rr = [r for r in rows if r["variant"] == v and r["event"].startswith(ev) and r["event"] not in ("E1(0.0)",)]
        if rr:
            w2 = sum(.5 <= r["ratio"] <= 2 for r in rr); w3 = sum(1 / 3 <= r["ratio"] <= 3 for r in rr)
            print(f"variant {v} {ev}: within 2x {w2}/{len(rr)}, within 3x {w3}/{len(rr)}")
