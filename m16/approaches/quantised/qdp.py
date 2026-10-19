"""T352 (4): exact forward DP for BINARY-activation models (README pre-registration).
P(correct) = sum_p int phi(v) F_p(v) dv, F_p(v) = P[exactly one x_q (q != p) > v, pick(h_10) = p | x_p = v]; DP over
(state, c = #inputs above v so far, capped at 2). Binary state h in {0,1}^16 -> 16-bit index. For a state, the input
line is cut at the breakpoints b_i = -(Wh h)_i / Wi_i into <= 17 intervals with fixed next states; Gaussian masses
from Phi at the breakpoints (float64 erfc, precomputed per state); splitting an interval at v uses min/max with
u = Phi(v), so the DP inner loop has no special functions. Reachable states: exact forward closure from h_0 = 0.
continuous inputs: the v-integral by Gauss-Legendre on M equal panels in u = Phi(v) (n nodes each); error estimated by
doubling M. discrete inputs (grid L): exact finite sum over levels (masses = Gaussian Voronoi-cell masses; ties: x_q
beats x_p iff level_q > level_p or (equal and q < p)).
FLOPs counted (T196: multiply-add = 2): DP updates + interval-mass ops; closure / tables counted separately.
usage: python qdp.py CKPT.pt NAME [M] [n] -> dp_<NAME>.json"""
import json, math, os, sys, time
import numpy as np, torch as th
from scipy.special import ndtr, ndtri
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.environ.get("QDEV", "cuda")


def closure(Wi, Wh):
    """exact reachable binary states per step; per state: interval breakpoints (sorted) and next states"""
    bits = 1 << np.arange(16); table = {}

    def trans(s):
        if s not in table:
            h = ((s >> np.arange(16)) & 1).astype(float); a = Wh @ h
            with np.errstate(divide="ignore", invalid="ignore"):
                b = np.where(Wi != 0, -a / Wi, np.nan)
            b = np.unique(b[np.isfinite(b)]); edges = np.r_[-np.inf, b, np.inf]
            mid = np.where(np.isinf(edges[:-1]), edges[1:] - 1, np.where(np.isinf(edges[1:]), edges[:-1] + 1, (edges[:-1] + edges[1:]) / 2))
            nxt = [int(((a + Wi * m_) > 0).astype(np.int64) @ bits) for m_ in mid]
            table[s] = (edges, nxt)
        return table[s]
    levels = [{0}]
    for t in range(10):
        nxt = set()
        for s in levels[-1]:
            nxt.update(trans(s)[1])
        levels.append(nxt)
    return levels, table


def tables(levels, table, Wo):
    U = sorted(set().union(*levels)); idx = {s: i for i, s in enumerate(U)}; S = len(U)
    J = max(len(table[s][1]) for s in U if s in table)
    PL = np.ones((S, J)); PR = np.ones((S, J)); NS = np.zeros((S, J), dtype=np.int64)
    for s in U:
        if s not in table:                                   # states reached only at t = 10: no outgoing transitions
            continue
        e, n = table[s]; k = len(n); i = idx[s]
        PL[i, :k] = ndtr(e[:-1]); PR[i, :k] = ndtr(e[1:]); NS[i, :k] = [idx[x] for x in n]
    H = np.array([((s >> np.arange(16)) & 1) for s in U], dtype=float); pick = (H @ Wo.T).argmax(1)
    return U, idx, PL, PR, NS, pick, J


def dp_cont(T_, Wi, Wh, U, idx, M, n, chunk=None):
    """continuous inputs: returns P(correct), FLOPs. M > 0: M equal panels in u; M = 0: EXACT mode, panels cut at every
    breakpoint Phi(b) of every reachable state (F_p is a polynomial of degree <= 9 in u on each panel, so n = 5
    Gauss-Legendre nodes per panel integrate it exactly)"""
    PL, PR, NS, pick = [th.as_tensor(a, device=DEV) for a in T_]; S, J = PL.shape
    xg, wg = np.polynomial.legendre.leggauss(n)
    if M > 0:
        e = np.linspace(0, 1, M + 1)
    else:
        e = np.unique(np.r_[0.0, T_[0][T_[0] > 0].ravel(), T_[1][T_[1] < 1].ravel(), 1.0])
    a_, b_ = e[:-1], e[1:]; keep = b_ > a_; a_, b_ = a_[keep], b_[keep]
    u = (a_[:, None] + (b_ - a_)[:, None] * (xg[None] + 1) / 2).ravel(); w = ((b_ - a_)[:, None] * wg[None] / 2).ravel()
    v = ndtri(u)
    Hm = np.array([((s >> np.arange(16)) & 1) for s in U], dtype=float); A = Hm @ Wh.T          # (S, 16)
    total = 0.0; flops = 0
    chunk = chunk or max(4, int(float(os.environ.get("DP_ELEMS", "3e7")) // (S * J * 3)))
    for c0 in range(0, len(u), chunk):
        uq = th.as_tensor(u[c0:c0 + chunk], device=DEV); wq = th.as_tensor(w[c0:c0 + chunk], device=DEV); Q = len(uq)
        vq = v[c0:c0 + chunk]
        # deterministic transition at t = p: next state of every state at input v
        nb = ((A[None] + vq[:, None, None] * Wi[None, None]) > 0).astype(np.int64) @ (1 << np.arange(16))   # (Q, S)
        NP = th.as_tensor(np.vectorize(lambda z: idx.get(int(z), -1))(nb), device=DEV)
        lo = th.clamp(th.minimum(PR[None], uq[:, None, None]) - PL[None], min=0)            # (Q, S, J) below v
        hi = th.clamp(PR[None] - th.maximum(PL[None], uq[:, None, None]), min=0)            # above v
        for p in range(10):
            d = th.zeros(Q, S, 3, dtype=th.float64, device=DEV); d[:, idx[0], 0] = 1.0
            for t in range(10):
                nd = th.zeros_like(d)
                if t == p:
                    ok = NP >= 0
                    nd.view(Q, S * 3).scatter_add_(1, (NP.clamp(min=0) * 3)[:, :, None].expand(Q, S, 3).reshape(Q, -1)
                                                   + th.arange(3, device=DEV).repeat(S)[None], (d * ok[..., None]).reshape(Q, -1))
                    flops += Q * S * 3
                else:
                    tgt = (NS * 3)[None, :, :, None] + th.arange(3, device=DEV)[None, None, None]       # (1, S, J, 3)
                    contrib_lo = d[:, :, None, :] * lo[..., None]                                     # stays c
                    contrib_hi = d[:, :, None, :] * hi[..., None]                                     # c -> min(c+1, 2)
                    up = th.tensor([1, 2, 2], device=DEV)
                    tgt_hi = (NS * 3)[None, :, :, None] + up[None, None, None]
                    nd.view(Q, -1).scatter_add_(1, tgt.expand(Q, S, J, 3).reshape(Q, -1), contrib_lo.reshape(Q, -1))
                    nd.view(Q, -1).scatter_add_(1, tgt_hi.expand(Q, S, J, 3).reshape(Q, -1), contrib_hi.reshape(Q, -1))
                    flops += Q * S * J * 3 * 4 + Q * S * J * 4
                d = nd
            F = (d[:, :, 1] * (pick == p)[None]).sum(1); total += float((F * wq).sum()); flops += Q * S * 2
        flops += Q * S * 16 * 2 * 2
    return total, flops


def dp_disc(Wi, Wh, Wo, G):
    """discrete inputs on G uniform levels of [-3, 3]: exact enumeration DP over (state, beats count)"""
    step = 6.0 / (G - 1); lv = -3.0 + step * np.arange(G); e = np.r_[-np.inf, (lv[:-1] + lv[1:]) / 2, np.inf]
    wl = ndtr(e[1:]) - ndtr(e[:-1]); bits = 1 << np.arange(16)
    levels = [{0}]; nxt_of = {}
    for t in range(10):
        new = set()
        for s in levels[-1]:
            if s not in nxt_of:
                h = ((s >> np.arange(16)) & 1).astype(float)
                nxt_of[s] = (((Wh @ h)[None] + lv[:, None] * Wi[None]) > 0).astype(np.int64) @ bits
            new.update(nxt_of[s].tolist())
        levels.append(new)
    U = sorted(set().union(*levels)); idx = {s: i for i, s in enumerate(U)}; S = len(U)
    NL = np.zeros((S, G), dtype=np.int64)
    for s, nx in nxt_of.items():
        NL[idx[s]] = [idx[int(z)] for z in nx]
    pick = (np.array([((s >> np.arange(16)) & 1) for s in U], dtype=float) @ Wo.T).argmax(1)
    NLt = th.as_tensor(NL, device=DEV); wt = th.as_tensor(wl, device=DEV); pk = th.as_tensor(pick, device=DEV)
    total = 0.0; flops = 0; lvl = th.arange(G, device=DEV)
    for p in range(10):
        for vi in range(G):                                            # x_p at level vi
            d = th.zeros(S, 3, dtype=th.float64, device=DEV); d[idx[0], 0] = 1.0
            for t in range(10):
                nd = th.zeros_like(d)
                if t == p:
                    nd.index_add_(0, NLt[:, vi], d); flops += S * 3
                else:
                    beat = (lvl > vi) | ((lvl == vi) & (t < p))           # (G,)
                    for c in range(3):
                        cn = th.where(beat, min(c + 1, 2), c)              # (G,)
                        src = d[:, c:c + 1] * wt[None]                     # (S, G)
                        nd.view(-1).index_add_(0, (NLt * 3 + cn[None]).reshape(-1), src.reshape(-1))
                    flops += S * G * 3 * 2
                d = nd
            total += wl[vi] * float((d[:, 1] * (pk == p)).sum()); flops += S * 2
    return total, flops, [len(l) for l in levels[1:]]


def main():
    f, name = sys.argv[1], sys.argv[2]; M = int(sys.argv[3]) if len(sys.argv) > 3 else 64; n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    d = th.load(f, weights_only=False); k = [sp["name"] for sp in d["meta"]["spec"]].index(name); sp = d["meta"]["spec"][k]
    assert sp["act"] == 1, "binary activations only"
    Wi, Wh, Wo = (d[a][k].double().numpy() for a in ("Wi", "Wh", "Wo")); t0 = time.time()
    levels, table = closure(Wi, Wh); counts = [len(l) for l in levels[1:]]
    T_ = tables(levels, table, Wo); U, idx = T_[0], T_[1]; print(name, "reachable per step", counts, "union", len(U), flush=True)
    out = dict(model=name, spec=sp, reachable=counts, union=len(U), max_intervals=T_[6], closure_sec=time.time() - t0, runs=[])
    if sp["grid"] == 0:
        for MM, nn in ((M, n), (2 * M, n), (4 * M, n)) if M > 0 else ((0, 5), (0, 6)):
            t1 = time.time(); P, fl = dp_cont(T_[2:6], Wi, Wh, U, idx, MM, nn)
            out["runs"].append(dict(M=MM, n=nn, P=P, flops=fl, sec=time.time() - t1)); print(name, out["runs"][-1], flush=True)
    else:
        t1 = time.time(); P, fl, lv = dp_disc(Wi, Wh, Wo, sp["grid"])
        out.update(reachable=lv, union=None); out["runs"].append(dict(exact_enumeration=True, P=P, flops=fl, sec=time.time() - t1))
        print(name, "reachable", lv, out["runs"][-1], flush=True)
    json.dump(out, open(os.path.join(HERE, f"dp_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
