"""T333 / addendum B (DIAGNOSTIC SAMPLING). B1: recovery of the product's MSE share by zero-cost rules (incl. ADD) in
V1's convention on 10 DEV2 models. B2: gate-pattern cell census of the joint (g12, g23) set at v2* per triple.
usage: python addB.py b1 | b2"""
import hashlib, json, os, sys
import numpy as np
which = sys.argv[1]; sys.argv = sys.argv[:1]
import budget as B  # noqa: E402
from budget import Point, mu, dmu, node, dev2, forward, T, TMAX, PTS, V4T  # noqa: E402


def build(P, v1, v2, v3, v4, low):
    x = np.empty((P.n, T)); r = np.arange(P.n)
    x[r, P.tri[:, 0]] = v1; x[r, P.tri[:, 1]] = v2; x[r, P.tri[:, 2]] = v3; x[r, P.p4] = v4
    for j in range(6):
        x[r, P.lows[:, j]] = low
    return x


def b1():
    tr = json.load(open(os.path.join(B.ROOT, "data/oracle_floor/dev2_truths.json")))["truths"]; out = {}
    models = dev2()
    for idx in range(0, 80, 8):
        name, model = models[idx]; rng = np.random.default_rng(324000 + idx); x = rng.standard_normal((B.N, T)); P = Point(x)
        v2a, g12s, g23s, v3a, v4a = B.anchor(P, True)
        v3 = v2a - P.g23; v4, low = B.closure_low(v3, v3a, v4a, True)
        L23 = forward(model, build(P, v2a + g12s, v2a, v3, v4, low))
        L12 = forward(model, build(P, v2a + P.g12, v2a, v3a, v4a, mu(v4a)))
        L0 = forward(model, build(P, v2a + g12s, v2a, v3a, v4a, mu(v4a)))
        LJ = forward(model, build(P, v2a + P.g12, v2a, v3, v4, low))
        j = P.tri[:, 1]; ok = lambda L: (L.argmax(1) == j).astype(float)
        s23, s12, J, A = ok(L23), ok(L12), ok(LJ), ok(L23 + L12 - L0)
        cut23 = P.g23 >= TMAX; cut12 = (P.g12 >= TMAX) | P.b
        comb = lambda X: np.where(cut23 & cut12, 1.0, np.where(cut23, s12, np.where(cut12, s23, X)))
        R = {"P1": comb(s23 * s12), "J1": comb(J), "ADD": comb(A), "OR": comb(np.maximum(s23, s12)), "L23": comb(s23), "L12": comb(s12)}
        out[name] = dict(truth=tr[name]["acc"], **{k: float(v.mean()) for k, v in R.items()},
                         agree_ADD_J=float((A == J).mean()), agree_P1_J=float((s23 * s12 == J).mean()))
        print(idx, name, {k: round(100 * (out[name][k] - out[name]["truth"]), 3) for k in R}, "agree ADD/J %.4f P1/J %.4f" % (out[name]["agree_ADD_J"], out[name]["agree_P1_J"]), flush=True)
    names = list(out); tru = np.array([out[n]["truth"] for n in names]); S = {}
    for k in ("P1", "J1", "ADD", "OR", "L23", "L12"):
        e = 100 * (np.array([out[n][k] for n in names]) - tru); S[k] = dict(bias=float(e.mean()), rms=float(np.sqrt((e ** 2).mean())), mse=float((e ** 2).mean()))
    share = S["P1"]["mse"] - S["J1"]["mse"]
    for k in S:
        S[k]["recovery"] = (S["P1"]["mse"] - S[k]["mse"]) / share
    S["go"] = bool(max(S[k]["recovery"] for k in ("ADD", "OR", "L23", "L12")) >= .4)
    json.dump(dict(per_model=out, summary=S), open(os.path.join(B.HERE, "addB1.json"), "w"), indent=1)
    for k, v in S.items():
        print(k, v if not isinstance(v, dict) else {a: round(b, 3) for a, b in v.items()})


def b2():
    """Distinct full gate patterns of the joint-set input over a grid in [0, tmax]^2, per triple and region."""
    import itertools
    TRI = np.array(list(itertools.permutations(range(T), 3))); models = dev2(); out = {}
    for idx, ng in ((0, 100), (40, 100), (72, 100), (0, 200)):
        name, (Wi, Wh, Wo) = models[idx]; g = (np.arange(ng) + .5) / ng * TMAX; G12, G23 = np.meshgrid(g, g, indexing="ij")
        G12 = G12.ravel(); G23 = G23.ravel(); res = {}
        for reg in ("nb", "b"):
            v2s, g12s, g23s = PTS[reg]; v3s = v2s - g23s; v4s = V4T[reg]
            counts = []
            for t in range(len(TRI)):
                tri = TRI[t]; free = [p for p in range(T) if p not in tri]; p4, lows = free[-1], free[:-1]
                n = len(G12); x = np.empty((n, T)); v3 = v2s - G23; v4 = v3 - (v3s - v4s)
                x[:, tri[0]] = v2s + G12; x[:, tri[1]] = v2s; x[:, tri[2]] = v3; x[:, p4] = v4
                x[:, lows] = (mu(np.array([v4s])) + dmu(np.array([v4s])) * (v4 - v4s))[:, None]
                h = np.zeros((n, 16)); bits = []
                for s in range(T):
                    h = np.maximum(h @ Wh.T + x[:, s:s + 1] * Wi[None], 0); bits.append(h > 0)
                pat = np.packbits(np.concatenate(bits, 1), axis=1)
                counts.append(len(np.unique(pat, axis=0)))
            res[reg] = dict(mean=float(np.mean(counts)), median=float(np.median(counts)), max=int(np.max(counts)), total=int(np.sum(counts)))
            print(name, ng, reg, res[reg], flush=True)
        out[f"{name}_{ng}"] = res
    json.dump(out, open(os.path.join(B.HERE, "addB2.json"), "w"), indent=1)


if __name__ == "__main__":
    b1() if which == "b1" else b2()
