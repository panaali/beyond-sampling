"""DEV-only predeclared full-mass matrix-rank falsifier; see README."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import json
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'm16'))
import leaderboard as lb
from approaches.sci_tao import core as C

OUT = Path(__file__).resolve().parent

def stats(actual, predicted):
    residual = predicted - actual
    threshold = np.quantile(actual, .95)
    tail = actual >= threshold
    den = max(float(np.linalg.norm(actual)), 1e-30)
    tailden = max(float(np.sum(actual[tail])), 1e-30)
    return dict(relative_l2=float(np.linalg.norm(residual) / den),
                true_mean=float(actual.mean()), predicted_mean=float(predicted.mean()),
                relative_mean_error=float(residual.mean() / max(actual.mean(), 1e-30)),
                top5_mass_recovery=float(predicted[tail].sum() / tailden),
                top5_absolute_error_mass=float(np.abs(residual[tail]).sum() / tailden),
                top5_true_mass_share=float(actual[tail].sum() / max(actual.sum(), 1e-30)),
                negative_fraction=float(np.mean(predicted < 0)))

def main():
    start = time.process_time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Optional cached public example checkpoint, no network')
    args = parser.parse_args()
    if args.checkpoint:
        import torch
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        base = tuple(state[k].detach().numpy().astype(np.float64) for k in
                     ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
        base = (base[0][:,0], base[1], base[2])
        models = {f'dev_s{sigma}_{k}': model for sigma in (.0025, .02)
                  for k, model in enumerate(lb._noisy(base, 6, sigma, seed=1))}
    else:
        models = lb.models(dev=True)
    report = dict(protocol='predeclared DEV full-mass rank falsifier', cases=[])
    arrays = {}
    skeletons = [((0,1,9), .5, .5), ((2,7,4), .8, .8)]
    for mi, name in enumerate(('dev_s0.0025_0', 'dev_s0.02_0')):
        for si, (cfg, tieq, maxq) in enumerate(skeletons):
            begin = time.process_time()
            rng = np.random.default_rng(290001 + si)  # same coordinates across DEV models
            rows, cols = rng.random((32,3)), rng.random((32,4))
            U = np.empty((32,32,10))
            ci = np.flatnonzero(np.all(C.configs(10, '23') == cfg, axis=1))[0]
            U[:,:,0] = (ci+.5)/len(C.configs(10, '23'))
            U[:,:,1] = tieq
            U[:,:,2] = maxq
            U[:,:,3:6] = rows[:,None,:]
            U[:,:,6:] = cols[None,:,:]
            y,a,b,tau = C.face_points(10, '23', U.reshape(-1,10))
            mass, optimistic, cells = C.walk_mass(models[name], y,a,b,tau, '23')
            A = mass.reshape(32,32)
            cellcost = 2*cells.reshape(32,32)
            assert np.all(np.isfinite(A)) and A.min() >= -1e-10
            assert cells.max() < 2000, 'walker hit cap'
            A = np.maximum(A,0)
            train = A[:16,:16]
            pivrow = qr(train.T, pivoting=True, mode='economic')[2]
            pivcol = qr(train, pivoting=True, mode='economic')[2]
            singular = np.linalg.svd(A, compute_uv=False)
            energy = np.cumsum(singular**2) / max(float(np.sum(singular**2)),1e-30)
            case = dict(model=name, skeleton=cfg, tie_quantile=tieq,
                        maximum_quantile=maxq, singular_values=singular.tolist(),
                        energy_rank={str(r):float(energy[r-1]) for r in (1,2,4,8,16)},
                        diagnostic_exact_faces=1024, diagnostic_walk_fpe=int(cellcost.sum()),
                        cells_mean=float(cells.mean()), cells_max=int(cells.max()),
                        true_mass_mean=float(A.mean()), true_mass_max=float(A.max()), curves=[])
            for rank in (2,4,8):
                ir,ic = pivrow[:rank],pivcol[:rank]
                pivot = A[np.ix_(ir,ic)]
                inv = np.linalg.pinv(pivot, rcond=1e-10)
                pred = A[:,ic] @ inv @ A[ir,:]
                observed = np.zeros_like(A,dtype=bool)
                observed[:16,:16]=True
                observed[:,ic]=True
                observed[ir,:]=True
                # Conservative dense arithmetic bound for both QR pivot searches,
                # pivot SVD/inverse, and full reconstruction, translated to FPE.
                algebra_flops_bound = 16*16**3 + 40*rank**3 + 2*32*rank**2 + 2*32**2*rank
                cost = float(cellcost[observed].sum() + algebra_flops_bound/6000)
                row = dict(rank=rank, construction_faces=int(observed.sum()),
                           construction_walk_fpe=int(cellcost[observed].sum()),
                           algebra_flops_bound=algebra_flops_bound,
                           construction_fpe_lower_bound=cost,
                           pivot_condition=float(np.linalg.cond(pivot)),
                           heldout=stats(A[16:,16:],pred[16:,16:]),
                           full_grid=stats(A,pred))
                # Integration comparison is CONDITIONAL on this fixed skeleton;
                # finite-grid discrepancy is not global accuracy or an MSE test.
                row['heldout']['mean_error_over_iid_se_same_faces'] = float(
                    (pred[16:,16:].mean()-A[16:,16:].mean()) /
                    max(A[16:,16:].std(ddof=1)/np.sqrt(observed.sum()),1e-30))
                case['curves'].append(row)
            case['cpu_seconds'] = time.process_time()-begin
            report['cases'].append(case)
            key = f'm{mi}_s{si}'
            arrays[key+'_mass']=A
            arrays[key+'_cells']=cells.reshape(32,32)
            arrays[key+'_rows']=rows
            arrays[key+'_cols']=cols
            report['cpu_seconds_total']=time.process_time()-start
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
            np.savez_compressed(OUT/'matrices.npz',**arrays)
            print(json.dumps(dict(model=name, skeleton=cfg, energy=case['energy_rank'],
                                  heldout=[dict(rank=r['rank'], **r['heldout']) for r in case['curves']],
                                  cpu_seconds=case['cpu_seconds'])),flush=True)
            if time.process_time()-start > 290:
                raise RuntimeError('CPU budget cutoff after saved case')
    print('TOTAL_CPU',time.process_time()-start,flush=True)

if __name__ == '__main__':
    main()
