"""Frozen DEV full-face replication of E15-family one-cell extrapolation.

Same pairwise node sets for exact and frozen-gate calculations; no fitted constants.
This is diagnostic fixed-node quadrature, not a new deductive estimator.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from probe import lb, tao
from envelope import decompose


def main():
    start = time.process_time()
    path = Path(__file__).parent
    state = torch.load('/workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth',
                       weights_only=True, map_location='cpu')
    base = (state['rnn.weight_ih_l0'].numpy()[:, 0].astype(float),
            state['rnn.weight_hh_l0'].numpy().astype(float), state['linear.weight'].numpy().astype(float))
    models = {f'dev_s{s}_0': lb._noisy(base, 6, s, seed=1)[0] for s in (.0025, .02)}
    _, z23, z12 = tao.tables(10)
    zs = {'23': z23, '12': z12}
    rows = []
    cells = 0
    for name, model in models.items():
        series = {}
        for ki, kind in enumerate(('23', '12')):
            # Predeclared fixed rules, same for both DEV models; prefix reuse.
            uniform = tao.sobol(10, 4096, 741033+ki)
            y, a, b, tau = tao.face_points(10, kind, uniform)
            d = np.zeros_like(y)
            ii = np.arange(len(y)); d[ii, a] = .5; d[ii, b] = -.5
            f = dict(x0=y, d=d, hi=np.sqrt(2)*tau)
            linear, count = decompose(model, f, linear=True)
            linerr = (linear.sum(1)-linear[:, 1])/(2*np.sqrt(np.pi))
            exact, _, counts = tao.walk_mass(model, y, a, b, tau, kind)
            cells += count+int(counts.sum())
            series[kind] = dict(linear=linerr, exact=exact, cells=counts)
        for n in (1024, 4096):
            bykind = {}
            for kind in ('23', '12'):
                a = series[kind]
                bykind[kind] = dict(exact_error=float(zs[kind]*a['exact'][:n].mean()),
                                    linear_error=float(zs[kind]*a['linear'][:n].mean()),
                                    exact_recurrence_fpe=int(2*a['cells'][:n].sum()),
                                    linear_recurrence_fpe=2*n)
            exact = sum(v['exact_error'] for v in bykind.values())
            linear = sum(v['linear_error'] for v in bykind.values())
            row = dict(model=name, nodes_per_kind=n, exact_node_error=exact,
                       linear_node_error=linear, accuracy_bias_vs_exact_nodes=exact-linear,
                       relative_error_mass_bias=linear/exact-1, terms=bykind)
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = dict(rows=rows, cpu_seconds=time.process_time()-start,
               recurrence_fpe_lower=2*cells,
               rule_seeds={'23':741033, '12':741034},
               note='Diagnostic fixed-node E15-family replication; paired exact-node integral is not ground truth; excludes algebra from FPE.')
    (path/'full_face_results.json').write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
