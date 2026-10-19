"""M3: locate and repair one far-gap rare error per candidate (generic, autograd; pre-registered in README section 3).

usage: python repair.py design TAG SEED    -> repair/<TAG>_s<SEED>_design.json (t0, located weights, edit)
       python repair.py verify TAG SEED    -> repair/<TAG>_s<SEED>_verify.json (targeted edit + 20 random controls)
Event E1(t0): the model picks the position of the 3rd largest while g23 = v2 - v3 > t0.
Slab sampler (conditional law given g23 > t0, as T410): v3 = b from a piecewise-constant grid proposal q with exact
weight G(b)/q(b), G(b) = 360 phi(b) Q(b + t0)^2 Phi(b)^7; two highs iid N | > b + t0, seven lows iid N | < b, uniform
random positions. P(E1) = E_q[w 1_E].
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch as th
from scipy.special import ndtr, ndtri

th.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import truth  # noqa: E402

OUT = os.path.join(HERE, "repair")
T0_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
_grid = np.linspace(-8, 5, 400_001)


def slab(t0, n, rng):
    """returns X (n, 10) float64, weight w (n,), pos3 (n,)."""
    b = _grid
    Gd = 360 * np.exp(-b ** 2 / 2) / np.sqrt(2 * np.pi) * (1 - ndtr(b + t0)) ** 2 * ndtr(b) ** 7
    cell = 0.5 * (Gd[1:] + Gd[:-1]) + 1e-300
    cdf = np.concatenate([[0], np.cumsum(cell)]); Z = cdf[-1]
    u = rng.random(n) * Z
    j = np.clip(np.searchsorted(cdf, u) - 1, 0, len(cell) - 1)
    v3 = b[j] + rng.random(n) * (b[1] - b[0])
    Gv = 360 * np.exp(-v3 ** 2 / 2) / np.sqrt(2 * np.pi) * (1 - ndtr(v3 + t0)) ** 2 * ndtr(v3) ** 7
    w = Gv * Z * (b[1] - b[0]) / cell[j]     # G / q, q = cell density / (Z * width)
    pl = ndtr(v3 + t0)
    hi = ndtri(np.clip(pl[:, None] + rng.random((n, 2)) * (1 - pl[:, None]), 1e-300, 1 - 1e-16))
    lo = ndtri(np.clip(rng.random((n, 7)) * ndtr(v3)[:, None], 1e-300, 1 - 1e-16))
    vals = np.concatenate([hi, v3[:, None], lo], 1)
    perm = np.argsort(rng.random((n, 10)), 1)              # perm[:, i] = position of value i
    X = np.empty((n, 10)); X[np.arange(n)[:, None], perm] = vals
    return X, w, perm[:, 2]


def params(m):
    return {k: v.detach().clone() for k, v in m.named_parameters()}


def flat_index(m):
    names, sizes = zip(*[(k, v.numel()) for k, v in m.named_parameters()])
    return list(names), np.cumsum((0,) + sizes)


def set_edit(m, base, idx, delta):
    names, off = flat_index(m)
    with th.no_grad():
        for k, v in m.named_parameters():
            v.copy_(base[k])
        flat = {k: v.view(-1) for k, v in m.named_parameters()}
        for i, dv in zip(idx, delta):
            j = np.searchsorted(off, i, side="right") - 1
            flat[names[j]][i - off[j]] += float(dv)


def count_events(models, t0, n, seed, chunk=250_000):
    """paired: per model, per 1e6-block sums of w 1_E; also P(g23 > t0) is the integral of G (not needed)."""
    rng = np.random.default_rng(seed); out = np.zeros((len(models), int(np.ceil(n / 1e6))))
    with th.no_grad():
        for i in range(0, n, chunk):
            X, w, p3 = slab(t0, min(chunk, n - i), rng)
            x = th.tensor(X)
            for k, m in enumerate(models):
                e = (m(x).argmax(-1).numpy() == p3)
                out[k, i // 1_000_000] += (w * e).sum()
    return out                                              # P = out.sum(1) / n


def acc_blocks(models, n, seed, chunk=250_000):
    rng = np.random.default_rng(seed); out = np.zeros((len(models), int(np.ceil(n / 1e6))))
    with th.no_grad():
        for i in range(0, n, chunk):
            x = th.tensor(rng.standard_normal((min(chunk, n - i), 10))); y = x.argsort(-1)[..., -2]
            for k, m in enumerate(models):
                out[k, i // 1_000_000] += (m(x).argmax(-1) == y).sum().item()
    return out


def design(tag, seed):
    m = truth.model(tag, seed); base = params(m); log = {"tag": tag, "seed": seed}; t_start = time.time()
    # t0: smallest grid value with pilot P <= 1e-6 and >= 50 pilot events (1e7 slab draws each)
    pilot = {}
    for t0 in T0_GRID:
        rng = np.random.default_rng(432000 + int(100 * t0)); n_ev = 0; s = 0.0
        with th.no_grad():
            for _ in range(40):
                X, w, p3 = slab(t0, 250_000, rng)
                e = m(th.tensor(X)).argmax(-1).numpy() == p3
                n_ev += int(e.sum()); s += (w * e).sum()
        pilot[t0] = (s / 1e7, n_ev)
        if s / 1e7 <= 1e-6 and n_ev >= 50:
            break
    ok = [t for t in pilot if pilot[t][0] <= 1e-6 and pilot[t][1] >= 50]
    if ok:
        t0, flag = ok[0], ""
    else:
        cand = [t for t in pilot if pilot[t][1] >= 50 and pilot[t][0] > 1e-6]
        t0 = min(cand, key=lambda t: pilot[t][0]) if cand else max(pilot, key=lambda t: pilot[t][1]); flag = "t0 rule fallback"
    log.update(pilot={str(k): v for k, v in pilot.items()}, t0=t0, t0_flag=flag)
    # locate: gradient of the smoothed event probability on 1e6 design draws
    rng = np.random.default_rng(432100); X, w, p3 = slab(t0, 1_000_000, rng)
    with th.no_grad():
        L = th.cat([m(th.tensor(X[i:i + 250_000])) for i in range(0, len(X), 250_000)])
        Lp = L[th.arange(len(L)), th.tensor(p3)]; Lo = L.clone(); Lo[th.arange(len(L)), th.tensor(p3)] = -th.inf
        marg = Lp - Lo.max(-1).values
    tau = 0.05 * marg.abs().median().item(); log["tau"] = tau
    m.zero_grad()
    for i in range(0, len(X), 250_000):
        x = th.tensor(X[i:i + 250_000]); pp = th.tensor(p3[i:i + 250_000]); ww = th.tensor(w[i:i + 250_000])
        L = m(x); Lp = L[th.arange(len(L)), pp]; Lo = L.masked_fill(Fn_onehot(pp, L), -th.inf)
        (ww * th.sigmoid((Lp - Lo.max(-1).values) / tau)).sum().div(len(X)).backward()
    g = th.cat([v.grad.view(-1) for _, v in m.named_parameters()]).numpy()
    S = np.argsort(-np.abs(g))[:12]
    names, off = flat_index(m)
    log["S"] = [{"flat": int(i), "param": names[np.searchsorted(off, i, side='right') - 1],
                 "index": int(i - off[np.searchsorted(off, i, side='right') - 1]), "g": float(g[i])} for i in S]
    log["g_norm"] = float(np.linalg.norm(g)); log["g_S_share"] = float(np.linalg.norm(g[S]) / np.linalg.norm(g))
    # accuracy curvature: Hessian of the CE loss restricted to S on 2^17 inputs
    xh = th.tensor(np.random.default_rng(432101).standard_normal((2 ** 17, 10))); yh = xh.argsort(-1)[..., -2]
    pnames = [k for k, _ in m.named_parameters()]
    def loss_of(delta):
        p = {k: v.clone() for k, v in base.items()}
        flat = {k: p[k].view(-1) for k in p}
        for j, i in enumerate(S):
            jj = np.searchsorted(off, i, side="right") - 1
            flat[names[jj]] = flat[names[jj]].index_add(0, th.tensor([int(i - off[jj])]), delta[j:j + 1])
        p = {k: flat[k].view(base[k].shape) for k in p}
        return th.nn.functional.cross_entropy(th.func.functional_call(m, p, (xh,)), yh)
    Hs = th.autograd.functional.hessian(loss_of, th.zeros(12, dtype=th.float64)).numpy()
    lam = 1e-3 * np.trace(Hs) / 12
    d = -np.linalg.solve(Hs + lam * np.eye(12), g[S]); d /= np.linalg.norm(d)
    log.update(H_eig=np.linalg.eigvalsh(Hs).tolist(), lam=lam, direction=d.tolist())
    set_edit(m, base, S, np.zeros(12))
    # step search on design draws (CRN)
    P0 = count_events([m], t0, 10_000_000, 432102).sum() / 1e7
    A0 = acc_blocks([m], 10_000_000, 432103).sum() / 1e7
    log.update(P0_design=P0, A0_design=A0)
    tried = {}
    def evaluate(eps):
        if eps not in tried:
            set_edit(m, base, S, eps * d)
            P = count_events([m], t0, 10_000_000, 432102).sum() / 1e7
            A = acc_blocks([m], 10_000_000, 432103).sum() / 1e7
            tried[eps] = (P, A)
        P, A = tried[eps]
        return P <= P0 / 20 and A - A0 >= -2e-4
    grid = [0.001 * 2 ** j for j in range(9)]
    first = next((e for e in grid if evaluate(e)), None)
    if first is not None:
        lo = first / 2 if first > grid[0] else 0.0; hi = first
        for _ in range(4):
            mid = 0.5 * (lo + hi)
            if evaluate(mid):
                hi = mid
            else:
                lo = mid
        eps, eflag = hi, ""
    else:
        okA = [e for e in tried if tried[e][1] - A0 >= -2e-4]
        eps = min(okA, key=lambda e: tried[e][0]) if okA else min(tried); eflag = "no eps met both design limits"
    log.update(eps=eps, eps_flag=eflag, tried={str(k): v for k, v in sorted(tried.items())},
               delta=(eps * d).tolist(), wall=time.time() - t_start,
               design_fpe=1e7 * len(pilot) + 4e6 + 2 ** 17 * 12 * 4 + 2e7 * (len(tried) + 1))  # fwd-equivalents (bwd ~ 2 fwd)
    os.makedirs(OUT, exist_ok=True)
    json.dump(log, open(os.path.join(OUT, f"{tag}_s{seed}_design.json"), "w"), indent=1)
    print(json.dumps({k: log[k] for k in ("t0", "t0_flag", "P0_design", "eps", "eps_flag")}), tried.get(eps))


def Fn_onehot(pp, L):
    return th.nn.functional.one_hot(pp, L.shape[-1]).bool()


def verify(tag, seed):
    D = json.load(open(os.path.join(OUT, f"{tag}_s{seed}_design.json")))
    t0, S = D["t0"], [s["flat"] for s in D["S"]]; delta = np.array(D["delta"]); nrm = np.linalg.norm(delta)
    base_m = truth.model(tag, seed); base = params(base_m)
    def edited(dv):
        m = truth.model(tag, seed); set_edit(m, base, S, dv); return m
    rng = np.random.default_rng(432300)
    rand = [z / np.linalg.norm(z) * nrm for z in rng.standard_normal((20, 12))]
    t_start = time.time()
    # targeted: 2e8 slab draws, 1e8 inputs (paired with base); controls: first 2e7 of each (same draws)
    allm = [base_m, edited(delta)] + [edited(r) for r in rand]
    ev_head = count_events(allm, t0, 20_000_000, 432200)
    ac_head = acc_blocks(allm, 20_000_000, 432201)
    ev_tail = count_events(allm[:2], t0, 180_000_000, 432210)
    ac_tail = acc_blocks(allm[:2], 80_000_000, 432211)
    ev = np.concatenate([ev_head[:2], ev_tail], 1); ac = np.concatenate([ac_head[:2], ac_tail], 1)
    def ratio_jk(e0, e1):
        r = e1.sum() / e0.sum(); B = len(e0)
        jk = np.array([(e1.sum() - e1[i]) / (e0.sum() - e0[i]) for i in range(B)])
        return r, float(np.sqrt((B - 1) / B * ((jk - jk.mean()) ** 2).sum()))
    r, rse = ratio_jk(ev[0], ev[1])
    dA = (ac[1] - ac[0]).sum() / 1e8; dse = float(np.std(ac[1] - ac[0], ddof=1) * np.sqrt(ac.shape[1]) / 1e8)
    res = {"tag": tag, "seed": seed, "t0": t0, "P_base": ev[0].sum() / 2e8, "ratio": r, "ratio_se": rse,
           "ratio_hi95": r + 1.96 * rse, "dacc": dA, "dacc_se": dse, "norm": nrm}
    res["success"] = bool(r <= 0.1 and r + 1.96 * rse <= 0.15 and dA >= -5e-4)
    ctrl = []
    for k in range(20):
        rk = ev_head[k + 2].sum() / ev_head[0].sum(); dk = (ac_head[k + 2] - ac_head[0]).sum() / 2e7
        ctrl.append({"ratio": rk, "dacc": dk, "success": bool(rk <= 0.1 and dk >= -5e-4)})
    res["controls"] = ctrl; res["n_ctrl_success"] = sum(c["success"] for c in ctrl)
    res["ctrl_min_ratio"] = min(c["ratio"] for c in ctrl)
    res["targeting_matters"] = bool(res["n_ctrl_success"] <= 2 and r < res["ctrl_min_ratio"])
    res["wall"] = time.time() - t_start
    json.dump(res, open(os.path.join(OUT, f"{tag}_s{seed}_verify.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "controls"}))


if __name__ == "__main__":
    {"design": design, "verify": verify}[sys.argv[1]](sys.argv[2], int(sys.argv[3]))
