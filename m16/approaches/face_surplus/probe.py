"""Frozen T34 hierarchical-surplus diagnostic; no accuracy fitting."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'm16'))
import leaderboard as lb
from approaches.sci_tao_v2 import core as C
from approaches.sci_tao_v2.compiled import walk_mass
OUT = Path(__file__).resolve().parent


def summaries(F, grid):
    rows = []
    for level in range(1, 9):
        stride = 2 ** (8-level)
        f = F[:, ::stride]
        x = grid[::stride][1::2]
        delta = f[:, 1::2] - (f[:, :-1:2] + f[:, 2::2])/2
        row = {'level': level}
        for tag, selected in [('all', np.ones(len(x), dtype=bool)),
                              ('interior', (x >= .05) & (x <= .95))]:
            d = delta[:, selected]
            energies = (d*d).mean(axis=1)
            count = max(1, int(np.ceil(.05 * len(energies))))
            row[tag] = dict(l1=float(np.abs(d).mean()), rms=float(np.sqrt(energies.mean())),
                            median_line_rms=float(np.median(np.sqrt(energies))),
                            top5_line_energy_share=float(np.sort(energies)[-count:].sum()/max(energies.sum(), 1e-300)))
        rows.append(row)
    exponents = {}
    for tag in ('all', 'interior'):
        for metric in ('l1', 'rms'):
            vals = np.array([r[tag][metric] for r in rows[3:]])
            exponents[tag+'_'+metric] = float(-np.polyfit(np.arange(4,9), np.log2(np.maximum(vals, 1e-300)), 1)[0])
    return dict(levels=rows, decay_exponents=exponents,
                mean_line_variance=float(np.var(F, axis=1).mean()),
                positive_fraction=float(np.mean(F > 0)))


def load(path):
    import torch
    s = torch.load(path, map_location='cpu', weights_only=True)
    w = tuple(s[k].detach().numpy().astype(np.float64) for k in
              ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
    return w[0][:,0], w[1], w[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint5', required=True)
    p.add_argument('--checkpoint10', required=True)
    args = p.parse_args()
    bases = {5:load(args.checkpoint5), 10:load(args.checkpoint10)}
    start = time.process_time()
    report = dict(protocol='T34 frozen dyadic surplus diagnostic', lines_per_coordinate=200,
                  levels=8, endpoint_clip=1e-5, cases=[], walker_fpe=0., faces=0)
    grid = np.linspace(1e-5, 1-1e-5, 257)
    for T, base in bases.items():
        for sigma in (.0025, .02):
            model = lb._noisy(base, 6, sigma, seed=1)[0]
            for family in ('23', '12'):
                rng = np.random.default_rng(340001 + T + int(family))
                anchors = rng.random((200, T))
                for axis in range(1, T):
                    U = np.repeat(anchors, 257, axis=0)
                    U[:, axis] = np.tile(grid, 200)
                    mass, costs, cells = [], [], []
                    for i in range(0, len(U), 4096):
                        y,a,b,tau = C.face_points(T, family, U[i:i+4096])
                        f,c,k = walk_mass(model,y,a,b,tau,family)
                        mass.append(f); costs.append(c); cells.append(k)
                    F = np.concatenate(mass).reshape(200,257)
                    row = dict(T=T, sigma=sigma, family=family, axis=axis,
                               face_evaluations=len(U), walker_fpe=float(np.concatenate(costs).sum()),
                               max_cells=int(np.concatenate(cells).max()), **summaries(F, grid))
                    report['cases'].append(row)
                    report['faces'] += len(U)
                    report['walker_fpe'] += row['walker_fpe']
                    report['cpu_seconds'] = time.process_time()-start
                    (OUT/'results.json').write_text(json.dumps(report, indent=2)+'\n')
                    print(json.dumps({k:row[k] for k in ('T','sigma','family','axis','decay_exponents','mean_line_variance')}), flush=True)
                    if report['cpu_seconds'] > 300:
                        raise RuntimeError('Diagnostic CPU cap reached; partial results saved')
    print('TOTAL', report['faces'], report['walker_fpe'], report['cpu_seconds'], flush=True)


if __name__ == '__main__':
    main()
