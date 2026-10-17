"""T64 isolated outer triangle driver; no canonical estimator modifications."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from stable_volume import adaptive_mass, chebyshev_tail
from cone_estimator import load_geometry


def integrate(rays, triangle, deadline, tolerance=1e-8, batch=128):
    start = time.process_time()
    groups = {}
    outer_bound = 0.
    for r in rays:
        gram = r.T @ r
        a = max(.25, float(gram.min()))
        b = max(a, float(np.diag(gram).max()))
        for n in (2, 3, 4, 6, 8, 12, 16, 20, 24):
            per_det = chebyshev_tail(a, b, n-2)/(6*np.pi**2)
            if per_det <= tolerance/16:
                break
        else:
            raise ArithmeticError('order ladder exhausted')
        det = abs(float(np.linalg.det(r)))
        outer_bound += det*per_det
        groups.setdefault(n, []).append((r, det))
    value = 0.
    inner_bound = 0.
    counters = {}
    sums = ('triangles', 'analytic_triangles', 'positive_quadrature_fallbacks',
            'quadrature_nodes', 'tangent_basis_constructions', 'great_circle_edges',
            'great_circle_arctan2_calls', 'solid_angle_arctan2_calls',
            'numerical_screen_rejections', 'bound_trials', 'bound_lgamma_calls',
            'bound_exp_calls', 'bound_log_calls', 'legendre_rule_generations',
            'legendre_rule_generation_nodes')
    for n, entries in groups.items():
        u, w = leggauss(n)
        u = (u+1)/2
        w = w/2*(1-u)**2
        for offset in range(0, len(entries), batch):
            if time.process_time() > deadline:
                raise TimeoutError('T64 cycle CPU cap')
            block = entries[offset:offset+batch]
            rr = np.stack([entry[0] for entry in block])
            dd = np.array([entry[1] for entry in block])
            vertices = (u[None, :, None, None]*rr[:, None, :, 0][:, :, None, :]
                        +(1-u)[None, :, None, None]*rr[:, None, :, 1:].transpose(0, 1, 3, 2))
            integrals, stats = triangle(vertices, exponent=4, absolute_tolerance=1e-11)
            weights = dd[:, None]/(2*np.pi**2)*w[None, :]
            value += float(np.sum(weights*integrals))
            # Aggregate fallback metadata has no individual weights. This
            # conservative maximum-weight bound remains valid for each block.
            inner_bound += float(weights.max())*(stats['quadrature_error_bound']
                            + stats.get('analytic_series_truncation_bound_sum', 0.))
            for key in sums:
                counters[key] = counters.get(key, 0)+stats.get(key, 0)
            key = 'maximum_heuristic_relative_roundoff'
            counters[key] = max(counters.get(key, 0.), stats.get(key, 0.))
    return dict(estimate=value, outer_integration_bound=outer_bound,
                inner_integration_bound=inner_bound, integration_only_bound=outer_bound+inner_bound,
                integration_cpu=time.process_time()-start,
                order_counts={n: len(v) for n, v in groups.items()},
                previous_3d_nodes=sum(n**3*len(v) for n, v in groups.items()),
                stats=counters, roundoff_certified=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--triangle-dir', required=True)
    p.add_argument('--geometry', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mode', choices=('pilot', 'full'), default='pilot')
    p.add_argument('--seconds', type=float, default=110.)
    args = p.parse_args()
    start = time.process_time()
    deadline = start+args.seconds
    sys.path.insert(0, args.triangle_dir)
    from triangle_batch import signed_l1_triangles
    data = dict(mode=args.mode, source_sha256={})
    for name in ('triangle_batch.py', 'triangle.py', 'moments.py', 'positive.py'):
        path = Path(args.triangle_dir)/name
        data['source_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    triangle, stats = signed_l1_triangles(np.eye(4)[:3], exponent=4)
    data['triangle_control'] = dict(estimate=float(triangle), truth=3*np.pi/4, stats=stats)
    data['cone_control'] = integrate(np.eye(4)[None], signed_l1_triangles, deadline)
    data['cone_control']['truth'] = 1/16
    import torch
    weights = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    base = tuple(weights[k].numpy().astype(float) for k in
                 ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
    base = (base[0].ravel(), *base[1:])
    rng = np.random.default_rng(51001)
    models = [base]+[tuple(w+sig*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape)
                          for w in base) for sig in (.0025, .02)]
    reference = json.loads((HERE.parent/'cone_results.json').read_text())
    geometry, sha = load_geometry(args.geometry)
    data['source_sha256']['geometry.py'] = sha
    data['models'] = []
    for sigma, model, ref in zip((0., .0025, .02), models, reference['models']):
        model_start = time.process_time()
        g = geometry.Geometry(seconds=deadline-time.process_time())
        cells = g.gates(model)
        correct = g.correct(model, cells)
        rays = g.simplices(correct)
        geometry_cpu = time.process_time()-model_start
        if args.mode == 'pilot':
            rays = rays[:32]
        result = integrate(rays, signed_l1_triangles, deadline)
        result.update(sigma=sigma, geometry_cpu=geometry_cpu,
                      production_cpu=time.process_time()-model_start,
                      simplices=len(rays), geometry_counts=g.counts.copy())
        if args.mode == 'pilot':
            control_start = time.process_time()
            previous = [adaptive_mass(r) for r in rays]
            result.update(reference=sum(x[0] for x in previous),
                          previous_integration_cpu=time.process_time()-control_start)
        else:
            result['reference'] = ref['rules'][-1]['estimate']
        result['difference'] = result['estimate']-result['reference']
        data['models'].append(result)
        data['total_cpu'] = time.process_time()-start
        (HERE/(args.mode+'_results.json')).write_text(json.dumps(data, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        if args.mode == 'pilot':
            break


if __name__ == '__main__':
    main()
