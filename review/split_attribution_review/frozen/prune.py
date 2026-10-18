"""T195 phase 2: budgeted-refinement replay of recipe108 (deductive variant; estT.py itself unchanged).
Rule: before refining a cell at step t (entry of _finish) or scoring a final leaf (_cell_error), test its Gaussian mass
against tau; if below, do not refine: give it the TRUE network's class at its ray centroid (one deterministic point)
and add its mass bound to the certified error bound B.
  test 'cap':  m_up = Gaussian mass of the circular cone around the normalised ray centroid u with half-angle
               max_r angle(u, r) (contains the cell, so an upper bound): 1/2 I_{sin^2 th}((d-1)/2, 1/2); O(m d) ops.
               Wrong-centroid pruned cells still need their exact mass for the estimate (charged); B uses m_up.
  test 'mass': exact estT.mass of every tested cell (charged); B uses the exact mass.
Everything charged lives in the estimator Ops; oracle diagnostics (uncharged, optional): the unpruned recipe108 subtree
of each pruned cell -> its recipe error mass and whether it is single-correctness (wrong mass 0 or full).
usage: python prune.py H T seed tau cap|mass [oracle] -> prune/H{H}_T{T}_s{seed}_{test}_{tau}.json"""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(HERE, "..", "understand_m43")
sys.path[:0] = [HERE, U, os.path.join(U, "..", "arc_m43_repro"), os.path.join(HERE, "..", "..")]
import numpy as np
from scipy.special import betainc
import estT
from estT import _margin_split, _cell_error, STATS
from m43 import Ops

CFG = dict(tau=0.0, test="cap", oracle=False)
P = dict(n=0, mass=0.0, bound=0.0, err=0.0, oracle_err=0.0, oracle_single=0.0, wrong_centroid=0, by_step={})


def cap_bound(c, ops):
    d = c.R.shape[1]; u = c.R.sum(0); u /= np.linalg.norm(u); cs = float(np.min(c.R @ u)); ops.add(len(c.R) * (d + 1) + 2 * d + 30)
    if cs <= 0:
        return 1.0
    return 0.5 * float(betainc((d - 1) / 2, 0.5, max(0.0, 1 - cs * cs)))


def centroid_class(c, model, ops):
    Wi, Wh, Wo = model; T = Wo.shape[0]; H = len(Wi); x = c.R.mean(0); h = np.zeros(H)
    for t in range(T):
        h = np.maximum(Wh @ h + x[t] * Wi, 0)
    ops.add(T * H * (H + 1) + T * H + len(c.R) * T)
    return int(np.argmax(Wo @ h))


def try_prune(c, g, A, t, model, truth, n_true, eps, last):
    """returns None (refine normally) or the pruned cell's error contribution (charged ops in `ops` by caller)."""
    ops = CUR_OPS[0]
    if CFG["test"] == "cap":
        mb = cap_bound(c, ops)
    else:
        mb = estT.mass(c, ops)
    if mb >= CFG["tau"]:
        return None
    wrong = centroid_class(c, model, ops) != truth
    if wrong:
        m = mb if CFG["test"] == "mass" else estT.mass(c, ops)
    else:
        m = 0.0
    P["n"] += 1; P["bound"] += mb; P["err"] += m; P["wrong_centroid"] += int(wrong)
    bs = P["by_step"].setdefault(t, [0, 0.0]); bs[0] += 1; bs[1] += mb
    if CFG["oracle"]:
        st = dict(STATS); mt = estT.mass(c, Ops())
        oe = _cell_error(c, g, A, model[2], truth, Ops()) if last else estT._finish(c, g, A, t, model, truth, n_true, eps, Ops(), False)
        STATS.clear(); STATS.update(st)
        P["mass"] += mt; P["oracle_err"] += oe
        if not (oe < 1e-15 or oe > mt - 1e-12):
            P["oracle_single"] += mt         # uncertain (mixed-correctness) pruned mass
    return m


CUR_OPS = [None]


def _finish(c, g, A, t, model, truth, n_true, eps, ops, exact):
    r = try_prune(c, g, A, t, model, truth, n_true, eps, False)
    if r is not None:
        return r
    Wi, Wh, Wo = model
    H = len(Wi); T = Wo.shape[0]; E = np.eye(T)
    At = Wh @ (g[:, None] * A) + np.outer(Wi, E[t]); ops.add(H * int(g.sum()) * t)
    err = 0.0
    for c2, g2 in _margin_split(c, At, model, t, truth, n_true, eps, ops, False):
        if t < T - 1:
            err += _finish(c2, g2, At, t + 1, model, truth, n_true, eps, ops, exact)
        else:
            r = try_prune(c2, g2, At, T, model, truth, n_true, eps, True)
            err += r if r is not None else _cell_error(c2, g2, At, Wo, truth, ops)
    return err


def main():
    from zoo44 import zoo
    from leaderboard import fpe_flops
    H, T, seed = (int(v) for v in sys.argv[1:4]); CFG.update(tau=float(sys.argv[4]), test=sys.argv[5], oracle=len(sys.argv) > 6)
    model = zoo(H, T, seed); estT.MODE[0] = "recipe108"; estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0)
    halves = estT.halves(T); o = Ops(); CUR_OPS[0] = o; t0 = time.time(); err = 0.0
    E = np.eye(T)
    for h in halves:
        g1 = (h["sigma"] * model[0] > 0).astype(float); o.add(H)
        err += _finish(h["cone"], g1, np.outer(model[0], E[0]), 1, model, h["truth"], h["n_true"], 0.1, o, False)
    r = dict(H=H, T=T, seed=seed, tau=CFG["tau"], test=CFG["test"], acc=1 - err, ops=o.n, flops=2 * o.n,
             N=2 * o.n / fpe_flops(H, T), sec=time.time() - t0, retained_cells=STATS["cells"], resolved=STATS["resolved"],
             pruned=P["n"], bound=P["bound"], pruned_err=P["err"], wrong_centroid=P["wrong_centroid"],
             by_step={str(k): v for k, v in P["by_step"].items()})
    if CFG["oracle"]:
        r.update(pruned_mass=P["mass"], oracle_err=P["oracle_err"], bound_oracle_uncertain=P["oracle_single"])
    os.makedirs(os.path.join(HERE, "prune"), exist_ok=True)
    json.dump(r, open(os.path.join(HERE, "prune", f"H{H}_T{T}_s{seed}_{CFG['test']}_{CFG['tau']:g}.json"), "w"), indent=1)
    print({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items() if k != "by_step"}, flush=True)


if __name__ == "__main__":
    main()
