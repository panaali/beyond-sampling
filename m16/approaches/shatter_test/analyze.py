"""T239 analysis of out/*.npz -> summary.json (pre-registered metrics in README)."""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); Q = (0.9, 0.99, 0.999); EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 9.0)


def cover(w, q):
    w = np.sort(w[w > 0])[::-1]; c = np.cumsum(w) / w.sum(); return int(np.searchsorted(c, q) + 1)


def strata(z):
    u = np.sort(z["u"], 1)[:, ::-1]; d = np.minimum(u[:, 0] - u[:, 1], u[:, 1] - u[:, 2]) / np.sqrt(2); blind = u[:, 1] <= 0
    out = {}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (d >= lo) & (d < hi) & ~blind
        out[f"{lo}-{hi}"] = dict(cells=int(s.sum()), mass=float(z["mass"][s].sum()), dens=float(s.sum() / max(z["mass"][s].sum(), 1e-300)))
    out["blind"] = dict(cells=int(blind.sum()), mass=float(z["mass"][blind].sum()), dens=float(blind.sum() / max(z["mass"][blind].sum(), 1e-300)))
    near = (d < 0.05) & ~blind; far = (d >= 0.1) & ~blind
    out["near_over_far"] = float((near.sum() / z["mass"][near].sum()) / (far.sum() / z["mass"][far].sum()))
    return out


S = {}
for f in sorted(glob.glob(os.path.join(HERE, "out", "*.npz"))):
    k = os.path.basename(f)[:-4]; z = np.load(f); r = dict(cells=len(z["mass"]), strata=strata(z))
    if k.startswith("final"):
        e = z["err"]; c = z["mass"] - e
        r.update(err_total=float(e.sum()), N_corr={str(q): cover(c, q) for q in Q}, N_err={str(q): cover(e, q) for q in Q},
                 err_cells=int((e > 0).sum()))
    S[k] = r
    st = r["strata"]
    print(k, r["cells"], {q: (r["N_corr"][q], r["N_err"][q]) for q in r.get("N_corr", {})}, "near/far %.1f" % st["near_over_far"],
          {s: "%d/%.3g" % (v["cells"], v["mass"]) for s, v in st.items() if isinstance(v, dict)})
json.dump(S, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
