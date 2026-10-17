import json, time
import numpy as np
from .enrich import CachedOracle, support_probes
from .additive import Additive
from .check_neural import main as neural, HERE


def controls():
    start = time.process_time(); shape = (7, 3, 3, 3)
    points = list(np.ndindex(shape))
    weights = [np.arange(1, n+1, dtype=float)/sum(range(1, n+1)) for n in shape]
    mass = np.array([np.prod([weights[k][p[k]] for k in range(4)]) for p in points])
    x = np.meshgrid(*[np.linspace(.1, .9, n) for n in shape], indexing='ij')
    spike = np.zeros(shape); spike[3, 1, 1, 1] = 1
    tensors = dict(zero=np.zeros(shape), constant=np.ones(shape),
                   additive_positive=sum(x), additive_signed=x[0]+2*x[1]-3*x[2]+.2*x[3],
                   central_spike=spike)
    out = dict(cases=[])
    for name, tensor in tensors.items():
        cache = CachedOracle(lambda ps: np.array([tensor[p] for p in ps]))
        fit = Additive(cache, shape, weights)
        value = fit.integral(); report = fit.report(); pred = fit.predict(points)
        defect = abs(value-pred@mass)
        err = float(np.max(abs(pred-tensor.ravel())))
        assert defect < 1e-11 and fit.fiber_defect < 1e-11
        if name != 'central_spike': assert err < 1e-10
        assert set(cache.cache) >= set(support_probes(shape))
        out['cases'].append(dict(name=name, contraction_defect=defect, max_error=err,
                                 integral=value, truth=float(tensor.ravel()@mass), report=report))
    z = CachedOracle(lambda ps: np.zeros(len(ps))); Additive(z, shape, weights)
    unseen = next(p for p in points if p not in z.cache)
    a = CachedOracle(lambda ps: np.array([float(p == unseen) for p in ps])); fit = Additive(a, shape, weights)
    assert a.cache.keys() == z.cache.keys() and fit.integral() == 0
    out['adversarial_spike'] = dict(location=unseen, unique_queries=len(a.cache), estimate=0,
                                   truth=float(mass[points.index(unseen)]))
    out['cpu_seconds'] = time.process_time()-start
    (HERE/'additive_controls_results.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out))


if __name__ == '__main__':
    controls()
    neural(recipes=('residual_additive',), outname='additive_results.json')
