"""Resume the frozen base T6 integral; every process retains its cold costs."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from ..accounting import Audit
from .integration import MassEngine

HERE = Path(__file__).resolve().parent
RESULT = HERE / 'full_results.json'
SETUP_OPS = 875013555

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    start = time.process_time()
    planning = json.loads((HERE / 'planning_results.json').read_text())
    cache = HERE / 'projected_correct_rays.npz'
    planfile = HERE / 'plan_1e-06.npz'
    sources = {name: digest(HERE / name) for name in
               ('integration.py', 'triangle.py', 'jacobi.py', 'geometry.py', 'full_run.py')}
    sources.update(projected_cache=digest(cache), plan=digest(planfile))
    assert sources['projected_cache'] == planning['projected_cache_sha256']
    data = np.load(cache)
    rays, projection = data['rays'], data['projection_bounds']
    plan = np.load(planfile)
    orders, bounds = plan['orders'], plan['bounds']
    assert rays.shape == (42012, 6, 6) and len(orders) == len(bounds) == len(rays)
    if RESULT.exists():
        out = json.loads(RESULT.read_text())
        assert out['sources'] == sources, 'Resume source/cache/plan changed'
        assert not out.get('failure'), 'Failure requires review before resuming'
        assert out['done'] == sum(c['end'] - c['start'] for c in out['cycles'])
        assert all(c['start'] == (out['cycles'][i-1]['end'] if i else 0)
                   for i, c in enumerate(out['cycles']))
    else:
        out = dict(sources=sources, done=0, total=len(rays), cycles=[],
                   partial_mass=0., outer_bound=0., projection_bound=0.,
                   inner_bound=0., series_bound=0., roundoff_certified=False,
                   floating_geometry_certified=False, preprocessing_operations=SETUP_OPS,
                   author_reference_midpoint=.24815022292589073,
                   author_reference_conditional_bound=9.04235e-6,
                   fixed_inner_budget=1e-8, fixed_outer_plan=1e-6)
    if out['done'] == len(rays):
        print('Already complete; no work repeated', flush=True)
        return
    prior_cpu = sum(c['cpu_seconds'] for c in out['cycles'])
    if prior_cpu >= 450:
        raise RuntimeError('Cumulative CPU reassessment required')
    audit = Audit()
    # Validation of dimensions, counters and bound accumulation; hashing, file
    # transport and JSON are excluded from arithmetic, included in CPU.
    audit.add('resume_scalar_setup', 10000)
    engine = MassEngine(audit)
    cycle = dict(start=out['done'], end=out['done'], cpu_seconds=0., status='running')
    out['cycles'].append(cycle)

    def save(status):
        cycle.update(end=out['done'], cpu_seconds=time.process_time()-start,
                     status=status, cost=audit.report())
        parts, events = Counter(), Counter()
        for item in out['cycles']:
            parts.update(item['cost']['parts'])
            events.update(item['cost']['events'])
            assert not item['cost']['black_box_kernels']
        operations = sum(parts.values())
        out.update(integration_operations=operations,
                   total_operations=SETUP_OPS+operations,
                   operation_components=dict(parts), events=dict(events),
                   integration_cpu_seconds=sum(c['cpu_seconds'] for c in out['cycles']),
                   complete=out['done'] == len(rays))
        out['conditional_error_bound'] = sum(out[k] for k in
            ('outer_bound', 'projection_bound', 'inner_bound', 'series_bound'))
        if out['complete']:
            p = out['partial_mass']
            samples = out['total_operations'] / 78.
            out['sampling_affine_only_operations_per_draw'] = 78
            out['sampling_variance_at_estimated_p'] = p*(1-p)/samples
            out['own_conditional_bound_squared_over_sampling_variance'] = (
                out['conditional_error_bound']**2 / out['sampling_variance_at_estimated_p'])
            delta = abs(p-out['author_reference_midpoint'])
            out['difference_from_author_midpoint'] = delta
            reference_worst = delta+out['author_reference_conditional_bound']
            out['reference_interval_worst_squared_error_over_sampling_variance'] = (
                reference_worst**2/out['sampling_variance_at_estimated_p'])
        temporary = RESULT.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(out, indent=2)+'\n')
        temporary.replace(RESULT)

    save('running')
    try:
        while out['done'] < len(rays):
            elapsed = time.process_time()-start
            if elapsed >= min(92., 448.-prior_cpu):
                break
            index = out['done']
            value, error, series, stats = engine.one(
                rays[index], int(orders[index]), 1e-8/len(rays))
            if not np.isfinite([value, error, series]).all() or min(value,error,series)<0:
                raise ArithmeticError('Invalid cone contribution')
            out['partial_mass'] += value
            out['inner_bound'] += error
            out['series_bound'] += series
            out['outer_bound'] += float(bounds[index])
            out['projection_bound'] += float(projection[index])
            out['done'] += 1
            audit.add('resume_per_cone_control_and_accumulation', 100)
            if out['done'] % 64 == 0:
                save('running')
        save('complete' if out['done']==len(rays) else 'checkpoint')
    except BaseException as exc:
        out['failure'] = dict(index=out['done'], type=type(exc).__name__, message=str(exc))
        save('failed')
        raise
    print(json.dumps({k:out[k] for k in ('done','total','partial_mass',
        'conditional_error_bound','integration_cpu_seconds','integration_operations','complete')}), flush=True)

if __name__ == '__main__':
    main()
