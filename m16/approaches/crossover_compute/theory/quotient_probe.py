"""Diagnostic only: exact common-translation/positive-scale quotient."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import stdtr, stdtrit

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'm16'))
from leaderboard import models, forward


def centered_directions(n, T, seed):
    u = np.random.default_rng(seed).standard_normal((n, T))
    u -= u.mean(axis=1, keepdims=True)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def common_shift_mass(model, u, tail=1e-10, max_cells=2000):
    wi, wh, wo = model
    n, T = u.shape
    H = len(wi)
    df = T - 1
    scale = np.sqrt(df)
    K = float(stdtrit(df, 1-tail) / scale)
    target = np.argsort(u, axis=1)[:, -2]
    now = np.full(n, -K)
    mass = np.zeros(n)
    cells = np.zeros(n, dtype=int)
    active = np.arange(n)
    e = 1 / np.sqrt(T)
    cdf = lambda q: stdtr(df, q * scale)
    for _ in range(max_cells):
        if len(active) == 0:
            break
        # Select right-hand gates without moving the integration endpoints.
        tc = now[active] + 1e-10 * (1 + np.abs(now[active]))
        h = np.zeros((len(active), H)); dh = np.zeros_like(h)
        nxt = np.full(len(active), K)
        for s in range(T):
            z = h @ wh.T + (u[active, s] + tc*e)[:, None] * wi
            dz = dh @ wh.T + e*wi
            with np.errstate(divide='ignore', invalid='ignore'):
                roots = tc[:, None] - z/dz
            roots = np.where((roots > tc[:, None]) & np.isfinite(roots), roots, np.inf)
            nxt = np.minimum(nxt, roots.min(axis=1))
            gate = z > 0
            h = z * gate; dh = dz * gate
        logits = h @ wo.T; slopes = dh @ wo.T
        rr = np.arange(len(active)); k = target[active]
        c = logits[rr, k, None] - logits
        a = slopes[rr, k, None] - slopes
        with np.errstate(divide='ignore', invalid='ignore'):
            roots = tc[:, None] - c/a
        lo = np.maximum(now[active], np.where(a > 0, roots, -np.inf).max(axis=1))
        hi = np.minimum(nxt, np.where(a < 0, roots, np.inf).min(axis=1))
        bad_constant = ((a == 0) & ((c < 0) | ((c == 0) & (np.arange(T) < k[:, None])))).any(axis=1)
        mass[active] += np.where((hi > lo) & ~bad_constant, cdf(hi)-cdf(lo), 0)
        cells[active] += 1
        now[active] = nxt
        active = active[nxt < K]
    if len(active):
        raise RuntimeError('gate walker did not finish')
    return mass, cells, {'tail_mass_bound': 2*tail, 'q_limit': K}


def dense_mass(model, u, n=8192):
    T = u.shape[1]
    q = stdtrit(T-1, (np.arange(n)+.5)/n)/np.sqrt(T-1)
    result = []
    for v in u:
        x = v + q[:, None]/np.sqrt(T)
        result.append(float(np.mean(forward(model, x).argmax(axis=1) == np.argsort(v)[-2])))
    return np.array(result)


def main():
    wall = time.monotonic(); cpu = time.process_time()
    # Two antipodal directions exhaust the residual sphere when T=2.
    toy = (np.array([1., -1.]), np.zeros((2, 2)), np.eye(2))
    u2 = np.array([[1., -1.], [-1., 1.]])/np.sqrt(2)
    tm, tc, _ = common_shift_mass(toy, u2)
    assert abs(tm.mean()-.75) < 2e-9, (tm, tc)
    # All exact logit ties must choose position 0.
    tie = (toy[0], toy[1], np.zeros((2, 2)))
    zm, _, _ = common_shift_mass(tie, u2)
    assert abs(zm.mean()-.5) < 2e-9, zm
    out = {'toy_accuracy': float(tm.mean()), 'toy_expected': .75,
           'zero_logits_accuracy': float(zm.mean()), 'models': []}
    all_models = models(dev=True)
    names = ['dev_s0.0025_0', 'dev_s0.02_0']
    for name in names:
        model = all_models[name]
        u = centered_directions(256, 10, 611)
        mw = time.monotonic(); mc = time.process_time()
        p, cells, meta = common_shift_mass(model, u)
        cpu_used = time.process_time()-mc; wall_used = time.monotonic()-mw
        exact_subset = p[:8]
        grid = dense_mass(model, u[:8])
        # Independent full-input validation. Its uncertainty dominates this small pilot.
        x = np.random.default_rng(712).standard_normal((50000, 10))
        raw = forward(model, x).argmax(axis=1) == np.argsort(x, axis=1)[:, -2]
        mean = float(p.mean()); variance = float(p.var(ddof=1))
        bernoulli = mean*(1-mean)
        record = dict(name=name, n_directions=len(u), mean=mean,
            conditional_variance=variance, bernoulli_variance=bernoulli,
            raw_variance_reduction=bernoulli/variance,
            mean_gate_cells=float(cells.mean()), max_gate_cells=int(cells.max()),
            min_gate_cells=int(cells.min()), actual_full_forward_fpe=int(2*cells.sum()),
            optimistic_cost_adjusted_gain=bernoulli/(variance*2*cells.mean()),
            dense_max_error=float(np.max(np.abs(exact_subset-grid))),
            dense_exact=exact_subset.tolist(), dense_grid=grid.tolist(),
            direct_accuracy=float(raw.mean()),
            diagnostic_z=float((mean-raw.mean())/np.sqrt(variance/len(u)+raw.var(ddof=1)/len(x))),
            validation_forward_evals=50000+8*8192,
            walker_cpu_seconds=cpu_used, walker_wall_seconds=wall_used, **meta)
        out['models'].append(record)
        print(json.dumps(record), flush=True)
    out.update(cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall,
               source_post_jan_2026_arc_read=False,
               classification='sampling diagnostic of exact analytic quotient; not a challenge submission')
    path = Path(__file__).with_name('quotient_results.json')
    path.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
