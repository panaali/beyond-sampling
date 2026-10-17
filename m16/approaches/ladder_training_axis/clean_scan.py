"""T101 preview (not a rung verdict): T75 and F2L3 on the 55 CLEAN models k = 30..20 x seeds 0-4.
usage: python clean_scan.py -> clean_scan.json"""
import json, os
import lt
out = {}
for k in range(30, 19, -1):
    for s in range(5):
        m = lt.zoo(k, s); tr, eo = lt.truth(m); r, o, st = lt.t75(m); f, fo, _ = lt.f2l(m)
        pq = tr * (1 - tr)
        d = dict(truth=float(tr), exact_ops=eo, t75_err=float(r - tr), t75_ops=o, t75_ratio=float((r - tr) ** 2 / (pq / (o / 72))),
                 f2l3_err=float(f - tr), f2l3_ops=fo, f2l3_ratio=float((f - tr) ** 2 / (pq / (fo / 72))), cells=st["cells"], resolved=st["resolved"], cut=st["cut"])
        out[f"{k}_{s}"] = d
        print(f"k={k} s={s} acc {tr:.4f} T75 err {100*d['t75_err']:+.3f}pp ratio {d['t75_ratio']:.3f} ops {o} | F2L3 err {100*d['f2l3_err']:+.1e}pp ratio {d['f2l3_ratio']:.1e} ops {fo}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_scan.json"), "w"), indent=1)
