"""Frozen cold neural construction, followed by separate full-grid diagnostics."""
import hashlib, json, time
from pathlib import Path
import numpy as np
from .enrich import CachedOracle, Enriched, cross, SOURCE, ROOT
from .additive import Additive

HERE = Path(__file__).resolve().parent
TRUTH = {2: .30773074302315223, 4: .47701312599404416}


def main(recipes=('original_rank2', 'original_rank4', 'residual_weighted', 'residual_unweighted'), outname='neural_results.json'):
    start = time.process_time()
    import torch
    # Explicitly use the reviewed Python backend; no optional native substitution.
    cross.WALK = cross.C.walk_mass
    paths = [SOURCE, ROOT/'m16/approaches/sci_tao_v2/core.py',
             ROOT/'m16/approaches/sci_tao/core.py', HERE/'enrich.py', HERE/'additive.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = dict(cases=[], sources=hashes, common_walker_dependency=True,
                  costs_are_proxies=True, validation_does_not_feed_construction=True)
    def save():
        result['cpu_seconds'] = time.process_time()-start
        (HERE/outname).write_text(json.dumps(result, indent=2)+'\n')
    for H in (2, 4):
        path = ROOT/f'work/checkpoints/H{H}_T4_seed0.pth'
        st = torch.load(path, map_location='cpu', weights_only=True)
        model = (st['rnn.weight_ih_l0'].numpy().astype(float).ravel(),
                 st['rnn.weight_hh_l0'].numpy().astype(float), st['linear.weight'].numpy().astype(float))
        unit = cross.C.fpe_flops(H, 4)
        _, z23, z12 = cross.C.tables(4)
        for order in (3, 5):
            case = dict(H=H, T=4, order=order, reference=TRUTH[H],
                        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), families=[])
            result['cases'].append(case)
            for kind, factor in (('23', z23), ('12', z12)):
                family = dict(kind=kind, factor=factor, recipes=[])
                case['families'].append(family)
                pending = []
                for name in recipes:
                    if time.process_time()-start > 105:
                        save(); raise TimeoutError('T81 cycle budget')
                    cold = time.process_time()
                    grid = cross.FaceGrid(model, kind, order)
                    cache = CachedOracle(grid)
                    if name.startswith('original'):
                        fit = cross.TensorCross(cache, grid.shape, int(name[-1]), 2)
                        integral = fit.integral(grid.weights)
                        report = dict(unique_queries=len(cache.cache), requests=cache.requests,
                                      original_cross_algebra_proxy=fit.flops, enrichment_work={})
                    elif name == 'residual_additive':
                        fit = Additive(cache, grid.shape, grid.weights)
                        integral = fit.integral()
                        report = fit.report()
                    else:
                        fit = Enriched(cache, grid.shape, grid.weights, weighted=name != 'residual_unweighted',
                                       local=name == 'residual_local')
                        integral = fit.integral()
                        report = fit.report()
                    elapsed = time.process_time()-cold
                    algebra = report['original_cross_algebra_proxy']+sum(report['enrichment_work'].values())
                    row = dict(recipe=name, integral=integral, construction_cpu_seconds=elapsed,
                               walk_fpe_proxy=grid.cost, algebra_proxy=algebra,
                               total_fpe_proxy=grid.cost+algebra/unit, cells=grid.cells, work=report)
                    family['recipes'].append(row)
                    pending.append((fit, row))
                # Only now enumerate the reference grid. No constructor has this array.
                validation_start = time.process_time()
                full = cross.FaceGrid(model, kind, order)
                points = np.array(list(np.ndindex(tuple(full.shape))))
                truth = full(points)
                weights = np.prod([full.weights[k][points[:, k]] for k in range(4)], axis=0)
                family.update(grid_integral=float(truth @ weights), grid_faces=len(points),
                              validation_walk_fpe_proxy=full.cost, validation_cells=full.cells)
                for fit, row in pending:
                    prediction = fit.predict(points)
                    row.update(grid_integral_error=row['integral']-family['grid_integral'],
                               weighted_l2=float(np.sqrt(np.sum(weights*(prediction-truth)**2))),
                               prediction_min=float(prediction.min()), prediction_max=float(prediction.max()),
                               negative_fraction=float(np.mean(prediction < 0)))
                family['validation_cpu_seconds'] = time.process_time()-validation_start
                save()
            case['full_grid_accuracy'] = 1-sum(f['factor']*f['grid_integral'] for f in case['families'])
            case['full_grid_bias'] = case['full_grid_accuracy']-TRUTH[H]
            case['recipes'] = []
            for k in range(len(recipes)):
                rows = [f['recipes'][k] for f in case['families']]
                accuracy = 1-sum(f['factor']*r['integral'] for f, r in zip(case['families'], rows))
                case['recipes'].append(dict(recipe=rows[0]['recipe'], accuracy=accuracy,
                    error_to_grid=accuracy-case['full_grid_accuracy'], error_to_reference=accuracy-TRUTH[H],
                    unique_queries=sum(r['work']['unique_queries'] for r in rows),
                    total_fpe_proxy=sum(r['total_fpe_proxy'] for r in rows),
                    construction_cpu_seconds=sum(r['construction_cpu_seconds'] for r in rows)))
            save()
            print(json.dumps({k:case[k] for k in ('H','order','full_grid_accuracy','recipes')}), flush=True)
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result['complete'] = True; save()


if __name__ == '__main__': main()
