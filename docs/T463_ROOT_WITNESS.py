"""Independent rational witness: six saved float upper bounds are too small.

No producer imports, interval library, model, RNG, or transcendental evaluation.
For x=g*(r+1+a,r+1,r,r-l_1,...,r-l_7), Jacobian is g**9.
On r in [-1/2,-1/4], g in [5/2,21/8], l_j in [0,1/8],
and the original thin a interval, ||x||^2 <= g_hi**2*qmax.
Since e<3 and pi<22/7, the Gaussian density is strictly greater
than (7/44)**5 / 3**ceil(g_hi**2*qmax/2). Multiply by the
minimum Jacobian and parameter volume for a rigorous rational mass lower bound.
"""
import json
from fractions import Fraction as F
from math import ceil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root / 'review/late_bound_soundness'
cases = {c['name']: c for c in json.loads((folder/'controls.json').read_text())['cases']}
saved = {r['name']: r for r in json.loads((folder/'first_pass/results.json').read_text())['mass']}
for k in (54, 55, 56, 60, 80, 100):
    name = f'thin_a_{k}'
    c = cases[name]
    assert list(map(F, c['lo'])) == [F(-3,4), F(1,2**k)] + [F(0)]*7
    assert list(map(F, c['hi'])) == [F(-1,4), F(1,2**(k-1))] + [F(1)]*7
    a0,a1 = F(c['lo'][1]), F(c['hi'][1])
    gl,gh = F(5,2), F(21,8)
    qmax = F(1,2)**2 + F(3,4)**2 + (F(3,4)+a1)**2 + 7*F(5,8)**2
    power = ceil(gh**2*qmax/2)
    lower = gl**9 * F(7,44)**5 * F(1,3**power) * F(1,4)*(a1-a0)*F(1,8)**7*(gh-gl)
    reported = F(saved[name]['float_upper'])
    assert reported > 0 and lower > reported
    assert 1.0 + float(a0) == 1.0 + float(a1) == 1.0
    print(name, 'mass_lower', float(lower), 'reported_upper', float(reported),
          'strict_rational_comparison', lower > reported)
