"""T424 scoring (pre-registered rules, README). Reads results/est/*.json and results/truth/*.json.
usage: python score.py -> results/score.json, results/score.txt"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY_1610 = ("m_16_10_cpu0", "m_16_10_zoo0", "m_16_10_zoo1")


def panel_of(tag, H, T, have):
    if (H, T) == (16, 10) and not all(f"m_16_10_cpu{s}" in have for s in (1, 2)):
        return "primary" if tag in PRIMARY_1610 else "supp"
    return "primary" if tag.endswith(("_cpu0", "_cpu1", "_cpu2")) else "supp"


def main():
    E = {json.load(open(f))["tag"]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "results", "est", "*.json"))}
    rows = []
    for tag, e in sorted(E.items(), key=lambda kv: (kv[1]["T"], kv[1]["H"], kv[0])):
        for t0 in e["t0s"]:
            fn = os.path.join(HERE, "results", "truth", f"{tag}@{t0:g}.json")
            tr = json.load(open(fn)) if os.path.exists(fn) else None
            r = dict(tag=tag, H=e["H"], T=e["T"], acc=e["acc"], t0=t0, panel=panel_of(tag, e["H"], e["T"], E))
            if tr is None or tr["hits"] < 50 or "rel_se" not in tr:
                r["truth"] = None if tr is None else ("infeasible" if "rel_se" in tr else "running"); rows.append(r); continue
            p = tr["p"]; r.update(truth=p, hits=tr["hits"], n=tr["n"], tie=tr["hits_g12_lt_p1"] / tr["hits"], in_range=1e-9 <= p <= 1e-6)
            for nm in ("k3", "graded"):
                if nm not in e:
                    continue
                est = e[nm]["est"][str(t0)]; C = e[nm]["C_fpe"]
                r[nm] = dict(est=est, ratio=est / p, ok=bool(.5 <= est / p <= 2), C=C, pC=p * C, F=(est - p) ** 2 * C / (p * (1 - p)))
            rows.append(r)
    pts = {}
    for r in rows:
        if r["panel"] != "primary" or not isinstance(r.get("truth"), float):
            continue
        pts.setdefault((r["H"], r["T"]), []).append(r)
    summ = {}
    for (H, T), rs in sorted(pts.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        s = dict(n=len(rs), tie_median=float(np.median([r["tie"] for r in rs])), p_range=[min(r["truth"] for r in rs), max(r["truth"] for r in rs)])
        for nm in ("k3", "graded"):
            q = [r for r in rs if nm in r]
            if not q:
                continue
            e2 = np.array([(r[nm]["est"] - r["truth"]) ** 2 for r in q]); pv = np.array([r["truth"] * (1 - r["truth"]) for r in q])
            Ceff = max(np.mean([r[nm]["C"] for r in q]), 1e3); Fp = float(e2.mean() * Ceff / pv.mean())
            k = sum(r[nm]["ok"] for r in q); inr = [r for r in q if r["in_range"]]
            s[nm] = dict(within2x=k, n=len(q), holds=bool(k >= 0.8 * len(q)), median_ratio=float(np.median([r[nm]["ratio"] for r in q])),
                         panel_F=Fp, strict=Fp <= 1, lenient=Fp <= 10, F_le1=sum(r[nm]["F"] <= 1 for r in q), F_le10=sum(r[nm]["F"] <= 10 for r in q),
                         in_range=f"{sum(r[nm]['ok'] for r in inr)}/{len(inr)}", C_median=float(np.median([r[nm]["C"] for r in q])))
        summ[f"{H},{T}"] = s
    json.dump(dict(rows=rows, summary=summ), open(os.path.join(HERE, "results", "score.json"), "w"), indent=1, default=float)
    L = ["| model | panel | t0 | truth | hits | tie share | K3 est/truth | graded est/truth | K3 F | graded F | K3 C | graded C |", "|" + "---|" * 12]
    for r in rows:
        if not isinstance(r.get("truth"), float):
            L.append(f"| {r['tag']} | {r['panel']} | {r['t0']:g} | {r['truth']} | | | | | | | | |"); continue
        g = lambda nm, k, fmt: (fmt % r[nm][k]) if nm in r else "-"
        L.append(f"| {r['tag']} | {r['panel']} | {r['t0']:g} | {r['truth']:.2e} | {r['hits']} | {r['tie']:.2f} | {g('k3','ratio','%.2f')} | "
                 f"{g('graded','ratio','%.2f')} | {g('k3','F','%.2g')} | {g('graded','F','%.2g')} | {g('k3','C','%.2e')} | {g('graded','C','%.2e')} |")
    L += ["", "| (H,T) | n | tie median | p range | K3 within 2x | K3 holds | K3 median | K3 panel F | graded within 2x | graded holds | graded median | graded panel F |", "|" + "---|" * 12]
    for k, s in summ.items():
        g = lambda nm: (f"{s[nm]['within2x']}/{s[nm]['n']} | {'yes' if s[nm]['holds'] else 'NO'} | {s[nm]['median_ratio']:.2f} | {s[nm]['panel_F']:.2g}") if nm in s else "- | - | - | -"
        L.append(f"| ({k}) | {s['n']} | {s['tie_median']:.2f} | {s['p_range'][0]:.1e}-{s['p_range'][1]:.1e} | {g('k3')} | {g('graded')} |")
    open(os.path.join(HERE, "results", "score.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
