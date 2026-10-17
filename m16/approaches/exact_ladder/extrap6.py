"""Cost/count extrapolation for T >= 6 without the full run (diagnostic, not an estimator).

Steps 1..T-1 are enumerated exactly (depth-first, counted). Of the step-(T-1) cells, every K-th (systematic
subsample, deterministic) is expanded to step T, split into outcome pieces and triangulated; every K2-th of those
also has its wrong simplices measured. Totals = (exact prefix part) + K x (subsample part) [+ K*K2 for measure].

usage: python extrap6.py T name K K2
"""
import json
import sys
import time

import numpy as np

sys.path.append("../..")
import leaderboard as L  # noqa: E402
import cones as E  # noqa: E402


def main(T, name, K, K2):
    t0 = time.process_time()
    m = L.models(16, T)[name]
    Wi = m[0]; H = len(Wi)
    pre = E.Ops(); last = E.Ops(); meas = E.Ops()
    counts = [0] * T
    roots = []
    for s in (1.0, -1.0):
        g = (s * Wi) > 0
        A = np.zeros((H, T)); A[:, 0] = np.where(g, Wi, 0.0)
        roots.append((np.array([[s]]), np.zeros((1, E.NW), np.uint64), A, E.bit(0)))
    counts[0] = 2
    stack = [(c, 1) for c in roots[::-1]]
    nsub = 0; nfin = 0; nw = 0; ng = 0; nsimp = 0; nsimp_meas = 0; wmass = 0.0; n_pen = 0
    base = 1 + 16 * (T - 1)
    while stack:
        c, t = stack.pop()
        if t == T - 1:
            n_pen += 1
            if n_pen % K:
                continue
            nsub += 1
            ch = E._children(*c, t, m, last)
            nfin += len(ch)
            for cc in ch:
                good, bad = E.outcome_pieces(cc, m, last, base)
                ng += len(good); nw += len(bad)
                V = []
                for R, Z in bad:
                    V += [R[list(tr)] for tr in E.triangulate(R, Z, last)]
                nsimp += len(V)
                if nsub % K2 == 0 and V:
                    p, _ = E.cone_measure(np.array(V), meas)
                    nsimp_meas += len(V); wmass += p.sum()
            if nsub % 200 == 0:
                print(f"  sub {nsub} pen {n_pen} counts {counts[:T-1]} cpu {time.process_time()-t0:.0f}", file=sys.stderr,
                      flush=True)
            continue
        ch = E._children(*c, t, m, pre)
        counts[t] += len(ch)
        stack += [(x, t + 1) for x in ch[::-1]]
    fpe = L.fpe_flops(16, T)
    est_cells = K * nfin
    est_meas_flops = meas.n * K * K2
    out = dict(T=T, name=name, K=K, K2=K2, cells_per_step_exact=counts[:T - 1], penultimate=n_pen, sampled=nsub,
               est_cells=est_cells, est_wrong_pieces=K * nw, est_correct_pieces=K * ng, est_simplices=K * nsimp,
               prefix_flops=pre.n, est_last_flops=K * last.n, est_measure_flops=est_meas_flops,
               est_total_fpe=(pre.n + K * last.n + est_meas_flops) / fpe,
               est_enum_fpe=(pre.n + K * last.n) / fpe, est_wrong_mass=wmass * K * K2,
               measured_simplices=nsimp_meas, cpu_s=time.process_time() - t0)
    print(json.dumps(out), flush=True)
    with open("extrap.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
