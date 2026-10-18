"""T333 B2' (DIAGNOSTIC): (1) gate-pattern cell census of J2's joint sets (4 half anchors x 720 triples) on 10 DEV2
models; (2) J2's error vs truth in V1's convention. usage: python addB2p.py census | rms"""
import itertools, json, os, sys
import numpy as np
which = sys.argv[1]; sys.argv = sys.argv[:1]
import budget as B  # noqa: E402
import addendum as A  # noqa: E402
from budget import Point, mu, dmu, node, dev2, forward, T, TMAX  # noqa: E402
TRI = np.array(list(itertools.permutations(range(T), 3)))
FREE = np.array([[p for p in range(T) if p not in t] for t in TRI]); P4 = FREE[:, -1]; LOWS = FREE[:, :-1]


def census(model, pt, ng, chunk=36):
    Wi, Wh, Wo = model; v2s, g12s, g23s = pt; v3s = v2s - g23s; v4s = float(node(np.array([v3s]))[0])
    g = (np.arange(ng) + .5) / ng * TMAX; G12, G23 = [a.ravel() for a in np.meshgrid(g, g, indexing="ij")]; m = len(G12)
    v3 = v2s - G23; v4 = v3 - (v3s - v4s); low = float(mu(np.array([v4s]))[0]) + float(dmu(np.array([v4s]))[0]) * (v4 - v4s)
    counts = np.zeros(len(TRI), int)
    for c0 in range(0, len(TRI), chunk):
        idx = np.arange(c0, min(c0 + chunk, len(TRI))); n = len(idx) * m
        x = np.empty((len(idx), m, T)); r = np.arange(len(idx))[:, None]
        x[r, :, TRI[idx, 0][:, None]] = (v2s + G12)[None]; x[r, :, TRI[idx, 1][:, None]] = v2s
        x[r, :, TRI[idx, 2][:, None]] = v3[None]; x[r, :, P4[idx][:, None]] = v4[None]
        for j in range(6):
            x[r, :, LOWS[idx, j][:, None]] = low[None]
        x = x.reshape(n, T); h = np.zeros((n, 16)); bits = []
        for s in range(T):
            h = np.maximum(h @ Wh.T + x[:, s:s + 1] * Wi[None], 0); bits.append(h > 0)
        pk = np.packbits(np.concatenate(bits, 1), axis=1)                      # (n, 20) bytes
        key = np.c_[np.repeat(idx, m)[:, None].astype(np.uint8).view(np.uint8) if False else np.zeros((n, 0), np.uint8), pk]
        h1 = pk[:, :8].copy().view(np.uint64)[:, 0]; h2 = pk[:, 8:16].copy().view(np.uint64)[:, 0]; h3 = np.zeros(n, np.uint64); h3[:] = pk[:, 16:20].copy().view(np.uint32)[:, 0]
        tid = np.repeat(idx, m).astype(np.uint64)
        u = np.unique(np.c_[tid, h1, h2, h3], axis=0)
        counts[idx] = np.bincount(u[:, 0].astype(int), minlength=len(TRI))[idx]
    return counts


if __name__ == "__main__":
    models = dev2()
    if which == "census":
        out = {}
        jobs = [(i, 100) for i in range(0, 80, 8)] + [(0, 200)]
        for idx, ng in jobs:
            name, model = models[idx]; res = {}
            for k, pt in A.PT2.items():
                c = census(model, pt, ng); res[k] = dict(total=int(c.sum()), mean=float(c.mean()), median=float(np.median(c)), max=int(c.max()))
            out[f"{name}_{ng}"] = res
            print(name, ng, {k: (v["total"], round(v["mean"], 1), v["max"]) for k, v in res.items()}, flush=True)
            json.dump(out, open(os.path.join(B.HERE, "addB2p_census.json"), "w"), indent=1)
    else:
        tr = json.load(open(os.path.join(B.ROOT, "data/oracle_floor/dev2_truths.json")))["truths"]; out = {}
        for idx in range(0, 80, 8):
            name, model = models[idx]; rng = np.random.default_rng(324000 + idx); x = rng.standard_normal((B.N, T)); P = Point(x)
            t23, t12, J2 = A.lines(model, P, A.anchor_from(P, A.PT2, True))
            cut23 = P.g23 >= TMAX; cut12 = (P.g12 >= TMAX) | P.b
            Va = np.where(cut23 & cut12, 1.0, np.where(cut23, t12, np.where(cut12, t23, J2)))
            out[name] = dict(truth=tr[name]["acc"], J2_v1conv=float(Va.mean()), J2_nocut_blindwalked=float(J2.mean()))
            print(idx, name, {k: round(100 * (v - out[name]["truth"]), 3) for k, v in out[name].items() if k != "truth"}, flush=True)
        names = list(out); tru = np.array([out[n]["truth"] for n in names]); S = {}
        for k in ("J2_v1conv", "J2_nocut_blindwalked"):
            e = 100 * (np.array([out[n][k] for n in names]) - tru); S[k] = dict(bias=float(e.mean()), rms=float(np.sqrt((e ** 2).mean())))
        print(S); json.dump(dict(per_model=out, summary=S), open(os.path.join(B.HERE, "addB2p_rms.json"), "w"), indent=1)
