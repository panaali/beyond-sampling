import json, time, resource
from pathlib import Path
import numpy as np
from fixed_tangent import fixed_tangent_tetra
from primitive import inverse_fifth_tetra
from outer import rule

BASE = Path(__file__).parent
start = time.process_time()
out = {'cases': []}
cases = [('identity', np.eye(5)[None])]
for name in ('t5_cones', 't5_sigma0.0025_cones', 't5_sigma0.02_cones'):
    rays = np.load(BASE.parent/(name+'.npz'))['rays']
    ids = np.linspace(0, len(rays)-1, 32, dtype=int)
    cases.append((name, np.swapaxes(rays[ids], 1, 2)))
p = np.array([.8, .05, .05, .05, .05])
for exponent in range(2, 9):
    rays = np.vstack([p, p + 10.**-exponent*(np.eye(5)[:4]-p)])
    cases.append(('asymmetric_'+str(exponent), rays[None]))
cases.append(('equal', np.tile(p, (1, 5, 1))))
for name, rays in cases:
    for order in (8, 16):
        u, w = rule(order)
        for variant in ('original', 'permuted_reflected'):
            v = rays if variant == 'original' else rays[:, [0, 4, 2, 1, 3]]*np.array([-1, 1, -1, 1, -1])
            endpoints = u[None, :, None, None]*v[:, 0, None, None] + (1-u[None, :, None, None])*v[:, None, 1:]
            t = time.process_time()
            try:
                actual, stats = fixed_tangent_tetra(v, u)
            except Exception as exc:
                failure = dict(name=name, order=order, variant=variant, error=repr(exc))
                (BASE/'fixed_controls_failure.json').write_text(json.dumps(failure, indent=2)+'\n')
                np.savez_compressed(BASE/'fixed_controls_failure.npz', rays=v, nodes=u)
                raise
            cpu = time.process_time()-t
            reference, refstats = inverse_fifth_tetra(endpoints, known_plane_normal=np.sign(v.sum(1))[:, None])
            relative = float(np.max(abs(actual-reference)/abs(reference)))
            error = stats.pop('quadrature_error_bounds')
            out['cases'].append(dict(name=name, order=order, variant=variant, maximum_relative_difference=relative,
                                     cpu_seconds=cpu, maximum_positive_quadrature_bound=float(error.max()), stats=stats,
                                     original_primitive_fallbacks=refstats['positive_fallbacks']))
            assert relative < 2e-9, out['cases'][-1]
out['cpu_seconds'] = time.process_time()-start
out['peak_rss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
(BASE/'fixed_controls.json').write_text(json.dumps(out, indent=2)+'\n')
print('cases',len(out['cases']),'max relative',max(c['maximum_relative_difference'] for c in out['cases']),'CPU',out['cpu_seconds'])
