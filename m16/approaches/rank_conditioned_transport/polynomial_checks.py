import json
import time
from collections import defaultdict
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.special import ndtr
from polynomial import interval_integrals, conditional_expectations
from engine import estimate

start = time.process_time()
mp.mp.dps = 70
result = dict(intervals=[], conditional_control={}, constant_controls=[])
stats = defaultdict(float)
left = np.array([-6., -6., -.1, .25, 5.5, 1., 0., -2.])
right = np.array([6., -5.999999999, .1, .250000000001, 6., 1.2, 1., -1.99])
mid, half = (left+right)/2, (right-left)/2
corners = tuple(np.tile([v, 1.], (len(left), 1)) for v in (.56, .32, .32, .8))
values, bounds = interval_integrals(left, right, corners, (mid+6)/12, (mid+6)/12,
                                   half/12, half/12, stats)
for i, (lo, hi) in enumerate(zip(left, right)):
    reference = float(mp.quad(lambda x: (mp.mpf('.5')+mp.mpf('.02')*x+mp.mpf('.005')*x*x)
                             *mp.exp(-x*x/2)/mp.sqrt(2*mp.pi), [mp.mpf(float(lo)), mp.mpf(float(hi))]))
    error = values[i, 0]-reference
    result['intervals'].append(dict(lo=float(lo), hi=float(hi), estimate=float(values[i, 0]),
                                    reference=reference, error=float(error), relative_error=float(error/reference),
                                    quadrature_only_bound=float(bounds[i])))
    assert abs(error/reference) < 2e-12

wi = np.ones(2)
wh = np.zeros((2, 2))
wo = np.zeros((3, 2))
grid = np.array([0., .1, .8, 2., 6.])
x, y = np.meshgrid(grid, grid, indexing='ij')
continuation = np.stack((.1+.02*x+.03*y+.004*x*y, np.ones_like(x)), axis=-1)
u = .35
q = (ndtr(u)-ndtr(-6))/(1-2*ndtr(-6))
lo, hi, error = conditional_expectations((wi, wh, wo), np.zeros((2, 2)), (grid, grid),
                                        continuation, u, q, 6., stats, start+18)
refs = []
for a, b in ((-6., u), (u, 6.)):
    integrand = lambda z: (mp.mpf('.1')+mp.mpf('.05')*max(z, 0)+mp.mpf('.004')*max(z, 0)**2)*mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)
    points = [a, 0., b] if a < 0 < b else [a, b]
    numerator = mp.quad(integrand, points)
    denominator = (mp.erf(b/mp.sqrt(2))-mp.erf(a/mp.sqrt(2)))/2
    refs.append(float(numerator/denominator))
result['conditional_control'] = dict(lower=lo.tolist(), upper=hi.tolist(), references=refs,
                                     maximum_error=float(max(abs(lo[0, 0]-refs[0]), abs(hi[0, 0]-refs[1]))),
                                     quadrature_only_bound=error)
assert result['conditional_control']['maximum_error'] < 1e-12
assert max(np.max(abs(lo[:, 1]-1)), np.max(abs(hi[:, 1]-1))) < 1e-12

for T in (3, 7):
    model = (np.array([.7, -.4]), np.array([[.8, -.3], [.2, .9]]), np.zeros((T, 2)))
    value, work = estimate(model, hidden_order=5, target_order=8, mesh_power=2.,
                           target_rule='split_sign', transition='polynomial', seconds=18-(time.process_time()-start))
    result['constant_controls'].append(dict(T=T, estimate=value, exact=1/T, error=value-1/T, stats=work))
    assert abs(value-1/T) < 2e-12
result['primitive_work'] = dict(stats)
result['total_cpu'] = time.process_time()-start
Path(__file__).with_name('polynomial_checks_results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
