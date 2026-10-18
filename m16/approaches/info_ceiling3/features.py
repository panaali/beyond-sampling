"""T332: per-row outputs of T317e (cont_e.py, verbatim @01906d33) -> 30 per-model features, for 80 DEV2 + 5 LB3 seeds.
Reruns cont_e.est's loop with read-only hooks; asserts the total equals cont_e.est bitwise. Diagnostic."""
import json
import sys
import numpy as np
import cont_e as CE

sys.path.insert(0, "/home/user/lb3/m16")
import leaderboard as LB  # noqa: E402

BINS = [(0, 1), (2, 4), (5, 9)]                       # ages 9 - p, inclusive
AGE2, AGE3 = 9 - CE.TRI[:, 1], 9 - CE.TRI[:, 2]


def tie(model, x, dx, k):
    Wi, Wh, Wo = model
    Z, DZ, G, *_ = CE.first_cell(model, x, dx)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L, dL = hT @ Wo.T, dhT @ Wo.T
    r = np.arange(len(k)); Lr = L.copy(); Lr[r, k] = -np.inf; js = Lr.argmax(1)
    return L[r, k] - L[r, js], dL[r, k] - dL[r, js]


def rows(model):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = CE.TRI[:, 1]; tot = np.zeros(CE.NT); out = {}
    for t in CE.TABLES:
        nodes, w, dens = t["nodes"], t["wts"], t["dens"][None]
        x, dx = t["line23"]
        p23, _ = CE.walk_counted(model, x, dx, k, CE.TMAX)
        R, A, B, _ = CE.merge(p23, CE.NT, CE.TMAX)
        I23, _ = CE.profile(R, A, B, CE.NT, nodes, "23", CE.TMAX, t["phi0"])
        c23 = np.bincount(np.concatenate([p[0] for p in p23]), minlength=CE.NT)
        m23, s23 = tie(model, x, dx, k)
        if "line12" in t:
            x12, dx12 = t["line12"]
            p12, _ = CE.walk_counted(model, x12, dx12, k, CE.TMAX)
            R, A, B, _ = CE.merge(p12, CE.NT, CE.TMAX)
            I12, _ = CE.profile(R, A, B, CE.NT, nodes, "12", CE.TMAX, t["phi0"])
            c12 = np.bincount(np.concatenate([p[0] for p in p12]), minlength=CE.NT)
            m12, s12 = tie(model, x12, dx12, k)
            full12 = (1.0 - CE.ndtr(nodes))[None]
            out["12nb"] = dict(d=((full12 - I12) * I23 * dens) @ w, m=m12, s=s12, c=c12)
        else:
            I12 = np.repeat(t["blind12"][None], CE.NT, 0)
        tag = "23nb" if t["name"].startswith("nb") else "23b"
        out[tag] = dict(d=(I12 * (t["phi0"]["m8"][None] - I23) * dens) @ w, m=m23, s=s23, c=c23)
        tot += (I12 * I23 * dens) @ w
    return float(tot.mean()), out


def feats(out):
    f, names = [], []
    for lt in ("23nb", "23b", "12nb"):
        for nm, age in (("p2", AGE2), ("p3", AGE3)):
            for lo, hi in BINS:
                sel = (age >= lo) & (age <= hi)
                f.append(np.where(sel, out[lt]["d"], 0.0).mean()); names.append(f"def_{lt}_{nm}age{lo}-{hi}")
    for lt in ("23nb", "23b", "12nb"):
        f += [np.clip(out[lt]["m"], -3, 3).mean(), (out[lt]["m"] < 0).mean()]; names += [f"mclip_{lt}", f"mneg_{lt}"]
    for lt in ("23nb", "23b", "12nb"):
        f.append(np.clip(out[lt]["s"], -30, 30).mean()); names.append(f"slope_{lt}")
    for lt in ("23nb", "23b", "12nb"):
        f.append(out[lt]["c"].mean()); names.append(f"cells_{lt}")
    return np.array(f), names


if __name__ == "__main__":
    Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
    models = [(str(nm), (Z["Wi"][i], Z["Wh"][i], Z["Wo"][i])) for i, nm in enumerate(Z["names"])]
    truth = {k: v["acc"] for k, v in json.load(open("/home/user/dl/dev2/truths.json"))["truths"].items()}
    lbm = LB.models(); lbt = json.load(open("/home/user/lb3/" + LB.truths_path(16, 10)))
    for s in range(5):
        models.append((f"seed{s}", lbm[f"seed{s}"])); truth[f"seed{s}"] = lbt[f"seed{s}"]["acc"]
    X, E, Y, N = [], [], [], []
    for name, m in models:
        e, out = rows(m)
        e0, _ = CE.est(m)
        assert e == e0, (name, e, e0)
        f, fn = feats(out)
        X.append(f); E.append(e); Y.append(truth[name]); N.append(name)
        print(name, round(e, 5), round(truth[name], 5), flush=True)
    np.savez("results/features.npz", X=np.array(X), est=np.array(E), truth=np.array(Y), names=np.array(N),
             fnames=np.array(fn))
