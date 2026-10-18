"""T338: W-derived allocations of exact lows on top of closure c; node counts; T298-protocol oracle floors on DEV2.
Diagnostic sampling (the floors); the allocation itself uses W and model-free anchors only."""
import json
import sys
import numpy as np
from scipy.special import ndtr, ndtri
import cont_e as CE

T, NT, TRI, FREE = 10, CE.NT, CE.TRI, CE.FREE
TIDX = np.full((10, 10, 10), -1); TIDX[TRI[:, 0], TRI[:, 1], TRI[:, 2]] = np.arange(NT)
TAB = {("nb" if t["name"].startswith("nb") else "b"): t for t in CE.TABLES}
BASE_NODES, BUDGET = 2815, int(0.3 * 2815)
mu = lambda v: -CE._pdf(v) / np.maximum(ndtr(v), 1e-300)


def gauss2(v3):
    """2-node Gauss rule of N(0,1) | < v3 (Stieltjes on a fine grid, like cont_e.gauss_v4)."""
    v = np.linspace(v3 - 9, v3, 40001)[:-1]; w = CE._pdf(v); w /= w.sum()
    m1 = (w * v).sum(); var = (w * (v - m1) ** 2).sum(); m3 = (w * (v - m1) ** 3).sum()
    # orthogonal poly recursion for K=2: a0 = m1, b1 = var, a1 = m1 + m3/var
    J = np.array([[m1, np.sqrt(var)], [np.sqrt(var), m1 + m3 / var]])
    x, U = np.linalg.eigh(J)
    return x, U[0] ** 2


def anchors(r):
    t = TAB[r]; v2, g12, g23 = t["point"]
    return CE.inputs(v2 + g12, v2, v2 - g23, t["v4"], t["m4"], 0, 0, 0, 0)[0], v2 - g23


def margin(model, X, k):
    Wi, Wh, Wo = model
    h = np.zeros((len(X), Wh.shape[0]))
    for s in range(T):
        h = np.maximum(h @ Wh.T + X[:, s:s + 1] * Wi[None], 0)
    L = h @ Wo.T; r = np.arange(len(X)); Lk = L[r, k].copy(); L[r, k] = -np.inf
    return Lk - L.max(1)


def deltas(model, r):
    X0, v3 = anchors(r); k = TRI[:, 1]
    m0 = margin(model, X0, k)
    qs = ndtri(np.array([.1, .9]) * ndtr(v3))
    D = np.zeros((NT, T))
    for j in range(T):
        for q in qs:
            X = X0.copy(); X[:, j] = q
            D[:, j] = np.maximum(D[:, j], np.abs(margin(model, X, k) - m0))
    isfree = np.zeros((NT, T), bool); isfree[np.arange(NT)[:, None], FREE] = True
    return np.where(isfree, D, -1.0)


def count_nodes(rows_x, rows_d):
    inv = np.zeros(len(rows_x), int); n = 0
    for s in range(T):
        key = np.round(np.column_stack([inv, rows_x[:, s], rows_d[:, s]]), 12)
        _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel(); n += int(inv.max()) + 1
    return int(n)


def line_rows(r, line, alloc):
    """Expand a line's 720 rows: allocated triple -> 2 rows with x_j at the 2 Gauss nodes (d = 0); closure c on the
    rest (v4 moves to the latest remaining free position when j is the latest)."""
    t = TAB[r]; x, d = t[line]; xs, ds = [], []
    nodes, _ = gauss2(t["point"][0] - t["point"][2])
    for i in range(NT):
        j = alloc[i]
        if j < 0:
            xs.append(x[i]); ds.append(d[i]); continue
        xi, di = x[i].copy(), d[i].copy()
        if j == FREE[i, -1]:                      # v4 slot taken: v4 to the next latest free position
            j2 = FREE[i, -2]; xi[j2], di[j2] = x[i, j], d[i, j]
        for nd in nodes:
            xx, dd = xi.copy(), di.copy(); xx[j], dd[j] = nd, 0.0; xs.append(xx); ds.append(dd)
    return np.array(xs), np.array(ds)


def lines_of(r):
    return ["line23", "line12"] if r == "nb" else ["line23"]


def allocate(cands, r):
    """cands: list of (priority, triple, j) sorted desc; greedy with est. cost (T - j), one per triple; then exact
    node count with trimming."""
    alloc = np.full(NT, -1); used = 0; order = []
    for _, i, j in cands:
        if alloc[i] >= 0 or used + (T - j) > BUDGET:
            continue
        alloc[i] = j; used += T - j; order.append(i)
    while True:
        cnt = {ln: count_nodes(*line_rows(r, ln, alloc)) for ln in lines_of(r)}
        if max(cnt.values()) <= BASE_NODES + BUDGET or not order:
            return alloc, cnt
        for i in order[-10:]:
            alloc[i] = -1
        order = order[:-10]


def allocations(model):
    out = {}
    for r in ("nb", "b"):
        D = deltas(model, r); A = {}
        mech = sorted(((D[i, j] / (T - j), i, j) for i in range(NT) for j in FREE[i]), reverse=True)
        A["mech"] = allocate(mech, r)
        late = sorted(((FREE[i, -1], -i, i, FREE[i, -1]) for i in range(NT)), reverse=True)
        A["late"] = allocate([(a, i, j) for a, _, i, j in late], r)
        adj = []
        for i in range(NT):
            cand = [j for j in FREE[i] if min(abs(j - TRI[i, 1]), abs(j - TRI[i, 2])) == 1]
            if cand:
                adj.append((max(cand), -i, i, max(cand)))
        A["adj"] = allocate([(a, i, j) for a, _, i, j in sorted(adj, reverse=True)], r)
        best = np.array([FREE[i][np.argmax(D[i, FREE[i]])] for i in range(NT)])
        A["mech_nobudget"] = (best, {ln: count_nodes(*line_rows(r, ln, best)) for ln in lines_of(r)})
        out[r] = A
    return out


G2 = {}
_grid = np.linspace(-4, 4.5, 341)
_tab = np.array([np.concatenate(gauss2(v)) for v in _grid])            # nodes(2), weights(2)


def closure_points(x, A, variant, mode):
    """Returns list of (X, weight) closure inputs for the batch x (n, 10). mode: 'oracle' or 'k2'."""
    n = len(x); r_ = np.arange(n)
    P = np.argsort(-x, 1); s = -np.sort(-x, 1)
    ti = TIDX[P[:, 0], P[:, 1], P[:, 2]]; nb = s[:, 1] > 0
    S = np.full(n, -1)
    if variant != "c":
        S = np.where(nb, A["nb"][variant][0][ti], A["b"][variant][0][ti])
    Fr = FREE[ti]                                                         # (n, 7) ascending
    inR = Fr != S[:, None]
    vals = np.where(inR, x[r_[:, None], Fr], -np.inf)
    mx = vals.max(1)
    latestR = np.where(inR, Fr, -1).max(1)
    X = x.copy()
    lowv = mu(mx)
    X[r_[:, None], Fr] = np.where(inR, lowv[:, None], X[r_[:, None], Fr])
    X[r_, latestR] = mx
    if mode == "oracle" or variant == "c":
        return [(X, np.ones(n))]
    has = S >= 0
    tb = np.array([np.interp(s[:, 2], _grid, _tab[:, c]) for c in range(4)])
    outs = []
    for a in range(2):
        Xa = X.copy(); Xa[r_[has], S[has]] = tb[a][has]
        outs.append((Xa, np.where(has, tb[2 + a], 0.5)))
    return outs


def ok(model, X, k):
    return margin(model, X, k) > 0                                         # argmax == k (ties measure-zero)


if __name__ == "__main__":
    Z = np.load("/home/user/dl/dev2/weights.npz", allow_pickle=True)
    truth = {k: v["acc"] for k, v in json.load(open("/home/user/dl/dev2/truths.json"))["truths"].items()}
    idx = range(len(Z["names"])) if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]
    N, CH = 500_000, 50_000
    VAR = [("c", "oracle"), ("late", "oracle"), ("mech", "oracle"), ("adj", "oracle"), ("mech_nobudget", "oracle"),
           ("late", "k2"), ("mech", "k2")]
    res = {}
    for i in idx:
        name = str(Z["names"][i]); m = tuple(np.asarray(Z[w][i], float) for w in ("Wi", "Wh", "Wo"))
        A = allocations(m)
        rng = np.random.default_rng(338000 + i); acc = np.zeros(len(VAR)); acc2 = np.zeros(len(VAR))
        for _ in range(N // CH):
            x = rng.standard_normal((CH, 10)); k = np.argsort(-x, 1)[:, 1]
            o = ok(m, x, k)
            for v, (var, mode) in enumerate(VAR):
                oc = sum(w * ok(m, X, k) for X, w in closure_points(x, A, var, mode))
                dlt = oc - o; acc[v] += dlt.sum(); acc2[v] += (dlt ** 2).sum()
        b = acc / N; se = np.sqrt((acc2 / N - b ** 2) / N)
        res[name] = {"truth": truth[name], "b": dict(zip([f"{a}_{b_}" for a, b_ in VAR], b.tolist())),
                     "se": dict(zip([f"{a}_{b_}" for a, b_ in VAR], se.tolist())),
                     "nodes": {r: {v: A[r][v][1] for v in A[r]} for r in A},
                     "n_alloc": {r: {v: int((A[r][v][0] >= 0).sum()) for v in A[r]} for r in A},
                     "alloc_pos_hist": {r: {v: np.bincount(A[r][v][0][A[r][v][0] >= 0], minlength=10).tolist()
                                            for v in A[r]} for r in A}}
        print(name, {k_: round(v_ * 100, 3) for k_, v_ in res[name]["b"].items()},
              {r: {v: max(A[r][v][1].values()) for v in A[r]} for r in A}, flush=True)
    tag = "all" if len(idx) > 1 else str(idx[0])
    json.dump(res, open(f"results/mc_{tag}.json", "w"), indent=1)
