"""Per-rung table from ev_<tag>.json: per seed (clean + 4 noisy pooled) MSE ratio vs MC-1e7 truth (truth noise
p(1-p)/1e7 subtracted) and vs the Hopf reference, for the frozen level and the estT recipe. -> report.json"""
import json, sys, numpy as np
OPS_S = {"44": 96, "54": 140, "64": 192, "43k20": 72}; FROZ = {"44": "h64", "54": "h64", "64": "h64", "43k20": "h1024"}
out = {}
for tag in sys.argv[1:]:
    d = json.load(open(f"ev_{tag}.json")); r = {}
    for k in (FROZ[tag], "recipe"):
        rs = []
        for s in sorted(d):
            rows = d[s]["rows"]; p = np.array([q["mc"] for q in rows]); e = np.array([q[k] for q in rows])
            if np.isnan(e).any():
                continue
            pq = np.mean(p * (1 - p)); ops = np.mean([q[k + "_ops"] for q in rows])
            mse_mc = max(np.mean((e - p) ** 2) - pq / 1e7, 0); pr = np.array([q["ref"] for q in rows])
            rs.append(dict(seed=int(s), ratio_mc=mse_mc / (pq * OPS_S[tag] / ops), ratio_mc_raw=np.mean((e - p) ** 2) / (pq * OPS_S[tag] / ops), ratio_ref=np.mean((e - pr) ** 2) / (pq * OPS_S[tag] / ops),
                           ops=ops, acc=float(p[0])))
        r[k] = rs
        if rs:
            print(f"{tag} {k}: ratio(MC truth) " + " ".join(f"{q['ratio_mc']:.3g}" for q in rs) + " | raw(no subtraction) " + " ".join(f"{q['ratio_mc_raw']:.3g}" for q in rs) + " | ratio(ref) " +
                  " ".join(f"{q['ratio_ref']:.3g}" for q in rs) + f" | ops {np.mean([q['ops'] for q in rs])/1e6:.2f}M acc " + " ".join(f"{q['acc']:.2f}" for q in rs))
    out[tag] = r
json.dump(out, open("report.json", "w"), indent=0, default=float)
