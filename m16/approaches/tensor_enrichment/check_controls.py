"""Independent direct contraction and observable support counterexamples."""
import itertools, json, time
from pathlib import Path
import numpy as np
from .enrich import CachedOracle, Enriched

HERE = Path(__file__).resolve().parent


def main():
    start = time.process_time()
    shape = (7, 3, 3, 3)
    points = list(np.ndindex(shape))
    weights = [np.arange(1, n+1, dtype=float) / sum(range(1, n+1)) for n in shape]
    mass = np.array([np.prod([weights[k][p[k]] for k in range(4)]) for p in points])
    xx = np.meshgrid(*[np.linspace(.1, .9, n) for n in shape], indexing='ij')
    spike = np.zeros(shape); spike[3, 1, 1, 1] = 1
    cases = dict(zero=np.zeros(shape), constant=np.ones(shape),
                 rank2=np.prod([1+x for x in xx], axis=0)+.3*np.prod([2-x for x in xx], axis=0),
                 signed_rank2=np.prod([1+x for x in xx], axis=0)-.3*np.prod([2-x for x in xx], axis=0),
                 central_spike=spike)
    out = dict(cases=[])
    for name, tensor in cases.items():
        for weighted in (False, True):
            oracle = CachedOracle(lambda ps: np.array([tensor[p] for p in ps]))
            fit = Enriched(oracle, shape, weights, weighted=weighted)
            value = fit.integral(); report = fit.report()
            pred = fit.predict(points)
            direct = float(pred @ mass)
            defect = abs(direct-value)
            error = float(np.max(abs(pred-tensor.ravel())))
            assert defect < 1e-10 and error < 1e-9, (name, defect, error)
            assert all(r.get('fiber_residual_after', 0) < 1e-9 for r in fit.records)
            out['cases'].append(dict(name=name, weighted=weighted, integral=value,
                                     contraction_defect=defect, max_error=error, work=report))
    # Choose the adversarial location after observing only the zero-response path.
    # Repeating with that spike must preserve exactly the algorithm's transcript.
    z = CachedOracle(lambda ps: np.zeros(len(ps)))
    f = Enriched(z, shape, weights)
    unseen = next(p for p in points if p not in z.cache)
    adv = CachedOracle(lambda ps: np.array([float(p == unseen) for p in ps]))
    g = Enriched(adv, shape, weights)
    assert adv.cache.keys() == z.cache.keys() and g.integral() == 0
    out['adversarial_spike'] = dict(location=unseen, true_integral=float(mass[points.index(unseen)]),
                                   estimate=g.integral(), unique_queries=len(adv.cache), total_entries=len(points))
    out['cpu_seconds'] = time.process_time()-start
    (HERE/'controls_results.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(dict(cpu=out['cpu_seconds'], maximum_error=max(r['max_error'] for r in out['cases']),
                          adversarial=out['adversarial_spike'])))


if __name__ == '__main__': main()
