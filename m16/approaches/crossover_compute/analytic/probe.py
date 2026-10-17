"""DEV-only falsification probe, not an accuracy estimator. See README."""
import itertools
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'm16/approaches/tube'),
                str(ROOT / 'm16/approaches/sci_laplace'), str(ROOT / 'm16')]
import lp
import leaderboard as lb
spec = importlib.util.spec_from_file_location('taocore_analytic', ROOT / 'm16/approaches/sci_tao/core.py')
tao = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tao)

JVP_INPUTS = 0
_jvp = lp.TB.fwd_jvp


def counted_jvp(model, x, dx):
    global JVP_INPUTS
    JVP_INPUTS += len(x)
    return _jvp(model, x, dx)


lp.TB.fwd_jvp = counted_jvp


def full_mass(model, f):
    n = len(f['hi'])
    mass, _, cells = tao.walk_mass(model, f['x0'], f['tpos'], f['tneg'],
                                  f['hi']/np.sqrt(2), '23')
    return mass * 2*np.sqrt(np.pi), int(cells.sum())


def anchored_rule(model, cfg, u, mx, q):
    """Full-band anchored HDMR integrated with 1D/2D Gauss-Legendre rules."""
    nodes, weights = np.polynomial.legendre.leggauss(q)
    nodes, weights = (nodes+1)/2, weights/2
    points = [np.full(7, 0.5)]
    for j in range(7):
        for node in nodes:
            x = np.full(7, 0.5); x[j] = node; points.append(x)
    for j in range(7):
        for k in range(j+1, 7):
            for a, b in itertools.product(nodes, repeat=2):
                x = np.full(7, 0.5); x[j], x[k] = a, b; points.append(x)
    low = ndtri(np.array(points)*ndtr(u))
    n = len(low)
    f = lp.build23(np.repeat(cfg[None], n, 0), np.full(n, u), np.full(n, mx), low)
    mass, cells = full_mass(model, f)
    singles = mass[1:1+7*q].reshape(7, q) @ weights
    pairs = mass[1+7*q:].reshape(21, q, q)
    pair_integrals = np.einsum('ijk,j,k->i', pairs, weights, weights)
    return dict(order1=float(singles.sum()-6*mass[0]),
                order2=float(pair_integrals.sum()-5*singles.sum()+15*mass[0]),
                points=n, cells=cells)


def gradients(model, x, a, b):
    wi, wh, wo = model
    n, t = x.shape
    h = np.zeros((n, len(wi)))
    j = np.zeros((n, len(wi), t))
    for k in range(t):
        z = h @ wh.T + x[:, k:k+1] * wi
        j = np.einsum('ij,njk->nik', wh, j, optimize=True)
        j[:, :, k] += wi
        j *= (z > 0)[:, :, None]
        h = np.maximum(z, 0)
    w = wo[a] - wo[b]
    return h @ w, np.einsum('j,njk->nk', w, j)


def feats(u, pairs=False):
    # Normalized shifted Legendre basis, orthonormal for uniform low-input ranks.
    z = 2*u - 1
    v = np.polynomial.legendre.legvander(z, 5)
    v *= np.sqrt(2*np.arange(6)+1)
    fs = [np.ones(len(u)), *[v[:, j, d] for j in range(7) for d in range(1, 6)]]
    if pairs:
        fs += [v[:, j, d]*v[:, k, e] for j in range(7) for k in range(j+1, 7)
               for d in range(1, 4) for e in range(1, 4)]
    return np.column_stack(fs)


def poly(z):
    cols = [np.ones(len(z))]
    for degree in range(1, 4):
        cols += [np.prod(z[:, ix], axis=1)
                 for ix in itertools.combinations_with_replacement(range(z.shape[1]), degree)]
    return np.column_stack(cols)


def fit_predict(f, g, y):
    gram = f.T @ f
    gram.flat[::len(gram)+1] += 1e-6 * len(f)
    return g @ np.linalg.solve(gram, f.T @ y)


def main():
    start = time.process_time()
    state = torch.load('/workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth',
                       weights_only=True, map_location='cpu')
    base = tuple(state[k].numpy().astype(float) for k in
                 ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
    base = (base[0][:, 0], base[1], base[2])
    # Exact lb.models(dev=True) formula, avoiding online reload of identical cached base.
    models = {f'dev_s{s}_0': lb._noisy(base, 6, s, seed=1)[0] for s in (0.0025, 0.02)}
    rng = np.random.default_rng(613017)
    combo, tie, top, _ = lp.rand23(6, rng)
    n = 8192
    rows = []
    full_mass_cells = 0
    gradient_inputs = 0
    for name, model in models.items():
        for case, (cfg, u, mx) in enumerate(zip(combo, tie, top)):
            rest = np.array([j for j in range(10) if j not in cfg])
            uv = rng.random((2*n, 7))
            low = ndtri(uv * ndtr(u))
            f = lp.build23(np.repeat(cfg[None], 2*n, 0), np.full(2*n, u),
                           np.full(2*n, mx), low)
            margin, jac = gradients(model, f['x0'], int(cfg[0]), int(cfg[1]))
            gradient_inputs += 2*n
            grad = jac[:n, rest]
            e, v = np.linalg.eigh(grad.T @ grad / n)
            ec = np.linalg.eigvalsh(np.cov(grad, rowvar=False, bias=True))
            active = {}
            for k in (2, 3):
                z = (low - low[:n].mean(0)) @ v[:, -k:]
                z /= np.std(z[:n], axis=0)
                pred = fit_predict(poly(z[:n]), poly(z[n:]), margin[:n])
                active[k] = float(1 - np.mean((pred-margin[n:])**2)/np.var(margin[n:]))
            delta, _, cells = lp.delta_ab(model, f)
            exact, cc = full_mass(model, {k: val[n:] for k, val in f.items()})
            full_mass_cells += cc
            # Every other readout is the midpoint of the tied pair's readouts:
            # away from exact output ties argmax is now strictly one of that pair.
            pair_wo = np.repeat(((model[2][cfg[0]]+model[2][cfg[1]])/2)[None], 10, 0)
            pair_wo[cfg[:2]] = model[2][cfg[:2]]
            pair_exact, cc = full_mass((model[0], model[1], pair_wo),
                                       {k: val[n:] for k, val in f.items()})
            full_mass_cells += cc
            truth_w = lp.W(0, np.abs(delta[n:])).mean()
            fits = {}
            for pair in (False, True):
                ftrain, ftest = feats(uv[:n], pair), feats(uv[n:], pair)
                pred = fit_predict(ftrain, ftest, delta[:n])
                pred = np.clip(pred, -f['hi'][n:], f['hi'][n:])
                w = lp.W(0, np.abs(pred))
                fits['pair' if pair else 'additive'] = {
                    'features': ftrain.shape[1],
                    'offset_r2': float(1-np.mean((pred-delta[n:])**2)/np.var(delta[n:])),
                    'mean_abs_offset_mass': float(w.mean()),
                    'paired_mass_bias': float((w-lp.W(0, np.abs(delta[n:]))).mean()),
                    'paired_mass_bias_se': float(np.std(w-lp.W(0, np.abs(delta[n:])))/np.sqrt(n)),
                }
            anchored = {q: anchored_rule(model, cfg, u, mx, q) for q in (5, 9)}
            full_mass_cells += sum(x['cells'] for x in anchored.values())
            row = dict(model=name, case=case, config=cfg.tolist(), tie=float(u), top=float(mx),
                       grad_energy_2=float(e[-2:].sum()/e.sum()),
                       grad_energy_3=float(e[-3:].sum()/e.sum()),
                       centered_grad_energy_2=float(ec[-2:].sum()/ec.sum()),
                       centered_grad_energy_3=float(ec[-3:].sum()/ec.sum()),
                       active_cubic_r2=active, exact_abs_offset_mass=float(truth_w),
                       exact_full_band_mass=float(exact.mean()),
                       exact_pair_only_mass=float(pair_exact.mean()),
                       exact_full_band_mass_se=float(exact.std()/np.sqrt(n)),
                       anchored=anchored, fits=fits)
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = dict(rows=rows, cpu_seconds=time.process_time()-start, inputs=gradient_inputs,
               full_jacobian_fpe_upper=gradient_inputs*11,
               full_mass_cells=full_mass_cells, jvp_inputs=JVP_INPUTS,
               full_walk_fpe_lower=2*(full_mass_cells+JVP_INPUTS),
               models='Exact leaderboard DEV seed=1 perturbation rule on cached base; no test inference',
               note='Diagnostic model-example fitting, not a deductive estimator; polynomial algebra cost additional.')
    Path(__file__).with_name('results.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
