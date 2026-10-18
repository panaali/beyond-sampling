"""T305: per-class closure error rates (two replicas) on 10 DEV2 models; ANOVA and sharing schemes."""
import itertools
import json
import sys
import time
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import norm

Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
NAMES = [str(n) for n in Z["names"]]
IDX = list(range(0, 80, 8))
T = 10


def predict(W, X):
    Wi, Wh, Wo = W
    h = np.zeros((len(X), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + X[:, t:t + 1] * Wi[None], 0)
    return (h @ Wo.T).argmax(1)


def topk_values(n, k, rng):
    u = rng.uniform(size=(n, k))
    v = np.empty((n, k))
    prev = np.ones(n)
    for i in range(k):
        v[:, i] = ndtri(prev * u[:, i] ** (1.0 / (T - i)))
        prev = ndtr(v[:, i])
    return v


def class_rates(W, classes, K, closure, rng, batch=200):
    k = len(classes[0])
    rates = np.empty(len(classes))
    for b0 in range(0, len(classes), batch):
        cl = np.array(classes[b0:b0 + batch])                  # (B, k)
        B = len(cl)
        P = np.repeat(cl, K, axis=0)                           # (B*K, k)
        v = topk_values(B * K, k, rng)
        r = np.arange(B * K)
        free = np.array([[p for p in range(T) if p not in row] for row in cl])
        F = np.repeat(free, K, axis=0)                         # free positions in time order
        X = np.empty((B * K, T))
        for i in range(k):
            X[r, P[:, i]] = v[:, i]
        if closure == "R0":
            mu = -norm.pdf(v[:, 2]) / ndtr(v[:, 2])
            X[r[:, None], F] = mu[:, None]
            wrong = (predict(W, X) != P[:, 1]).astype(float)
        else:                                                  # T4S
            q = np.arange(6, 0, -1) / 7.0
            sp = ndtri(ndtr(v[:, 3:4]) * q[None])
            X1, X2 = X.copy(), X.copy()
            X1[r[:, None], F] = sp
            X2[r[:, None], F] = sp[:, ::-1]
            wrong = 1 - 0.5 * ((predict(W, X1) == P[:, 1]).astype(float) + (predict(W, X2) == P[:, 1]))
        rates[b0:b0 + B] = wrong.reshape(B, K).mean(1)
    return rates


def schemes(classes):
    k = len(classes[0])
    S = {}
    S["identity"] = (lambda c: c, min)
    S["shift_start"] = (lambda c: tuple(p - c[0] for p in c[1:]), lambda g: min(g, key=lambda c: c[0]))
    S["shift_end"] = (lambda c: tuple(p - c[0] for p in c[1:]), lambda g: max(g, key=lambda c: c[0]))
    S["share_last"] = (lambda c: c[:-1], lambda g: min(g, key=lambda c: c[-1]))
    S["share_p1"] = (lambda c: c[1:], lambda g: min(g, key=lambda c: c[0]))
    S["age_cap3"] = (lambda c: tuple(min(9 - p, 3) for p in c), min)
    S["pos_cap3"] = (lambda c: tuple(min(p, 3) for p in c), min)
    return S


def anova(classes, rbar, noise_var):
    C = np.array(classes)
    k = C.shape[1]
    tot = rbar.var()

    def r2(cols):
        X = np.concatenate([np.ones((len(C), 1))] + cols, 1)
        beta = np.linalg.lstsq(X, rbar, rcond=None)[0]
        return 1 - (rbar - X @ beta).var() / tot
    oh = [np.eye(T)[C[:, i]] for i in range(k)]
    out = {f"p{i + 1}_alone": r2([oh[i]]) for i in range(k)}
    out["main_all"] = r2(oh)
    pairs = []
    for i, j in itertools.combinations(range(k), 2):
        pairs.append(np.eye(T * T)[C[:, i] * T + C[:, j]])
    out["main+pairs"] = r2(oh + pairs)
    out["signal_share"] = float(max(tot - noise_var, 0) / tot)
    return {k_: float(v) for k_, v in out.items()}


closure = sys.argv[1]
k = 3 if closure == "R0" else 4
K = 2000 if closure == "R0" else 400
classes = [c for c in itertools.permutations(range(T), k)]
res = {}
for m_i in IDX:
    W = (Z["Wi"][m_i].astype(float), Z["Wh"][m_i].astype(float), Z["Wo"][m_i].astype(float))
    t0 = time.time()
    rA = class_rates(W, classes, K, closure, np.random.default_rng(305000 + m_i))
    rB = class_rates(W, classes, K, closure, np.random.default_rng(305500 + m_i))
    rbar = 0.5 * (rA + rB)
    noise_var = float(np.var(rA - rB) / 4)
    ix = {c: i for i, c in enumerate(classes)}
    row = {"closure_error_pp": 100 * float(rbar.mean()), "anova": anova(classes, rbar, noise_var), "schemes": {}}
    for sn, (key, pick) in schemes(classes).items():
        groups = {}
        for c in classes:
            groups.setdefault(key(c), []).append(c)
        e = 0.0
        for g in groups.values():
            rep = ix[pick(g)]
            e += sum(rA[rep] - rB[ix[c]] for c in g)
        row["schemes"][sn] = {"groups": len(groups), "reduction": len(classes) / len(groups),
                              "err_pp": 100 * e / len(classes)}
    res[NAMES[m_i]] = row
    print(NAMES[m_i], "err %.2f pp" % row["closure_error_pp"],
          {s: round(v["err_pp"], 3) for s, v in row["schemes"].items()}, "%.0fs" % (time.time() - t0), flush=True)
json.dump(res, open(f"results/{closure}.json", "w"), indent=1)
print("\nscheme: groups, reduction, RMS err pp over models")
for sn in res[NAMES[IDX[0]]]["schemes"]:
    e = np.array([res[n]["schemes"][sn]["err_pp"] for n in res])
    g = res[NAMES[IDX[0]]]["schemes"][sn]
    print(f"  {sn}: groups {g['groups']} reduction {g['reduction']:.1f}x RMS {np.sqrt((e ** 2).mean()):.3f} mean {e.mean():+.3f}")
A = {k_: np.mean([res[n]["anova"][k_] for n in res]) for k_ in res[NAMES[IDX[0]]]["anova"]}
print("ANOVA (mean over models, share of raw variance):", {k_: round(v, 3) for k_, v in A.items()})
