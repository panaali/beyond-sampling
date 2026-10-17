"""H2T4 outward mass check, exact setup controls and fixed arithmetic falsifiers."""
from fractions import Fraction as F
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time
import numpy as np
from exact import Exact
from enclosed_mass import EnclosedMass, PositiveIntervals, fraction_bounds

HERE = Path(__file__).resolve().parent


def arithmetic_controls():
    if np.finfo(np.float64).nmant != 52 or np.dtype(np.float64).itemsize != 8:
        raise ArithmeticError('binary64 required')
    if float(np.add(np.float64(1.), np.float64(2.**-53))) != 1.:
        raise ArithmeticError('round-to-nearest-even check failed')
    if float(np.add(np.float64(1.), np.float64(3*2.**-53))) != 1.+2.**-51:
        raise ArithmeticError('second tie rounding check failed')
    if float(np.multiply(np.float64(2.**-1022), np.float64(.5))) != 2.**-1023:
        raise ArithmeticError('gradual underflow check failed')
    if float(np.nextafter(np.float64(0.), np.float64(1.))) != float.fromhex('0x0.0000000000001p-1022'):
        raise ArithmeticError('nextafter/subnormal check failed')
    values = [0., float.fromhex('0x0.0000000000001p-1022'), 2.**-500,
              math.nextafter(1., 0.), 1., math.nextafter(1., math.inf), 2.**500]
    counts = defaultdict(int)
    ops = PositiveIntervals(counts)
    checked = 0
    for a in values:
        for b in values:
            candidates = [('add', ops.add, F.from_float(a)+F.from_float(b)),
                          ('multiply', ops.mul, F.from_float(a)*F.from_float(b))]
            if b and math.isfinite(a/b):
                candidates.append(('divide', ops.div, F.from_float(a)/F.from_float(b)))
            for name, function, exact in candidates:
                lo, hi = function((a, a), (b, b))
                if not F.from_float(float(lo)) <= exact <= F.from_float(float(hi)):
                    raise ArithmeticError(f'{name} exact Fraction containment failed')
                checked += 1
    for x in (F(0), F(1), F(1, 3), F(1, 2**1075), F(2**500+1, 2**500)):
        lo, hi = fraction_bounds(x)
        assert F.from_float(lo) <= x <= F.from_float(hi)
    return dict(fixed_elementary_operation_cases=checked, counts=dict(counts),
                binary64=True, tested_gradual_underflow=True, tested_round_to_nearest_even=True)


def load_exact_simplices():
    path = HERE/'geometry_t4.json'
    data = json.loads(path.read_text())
    exact = Exact(seconds=10)
    simplices = []
    for item in data['correct_simplices']:
        rays = tuple(tuple(map(int, row)) for row in item['integer_rays'])
        determinant = F(*map(int, item['normalized_absolute_determinant']))
        checked = F(abs(exact.det(rays)), math.prod(sum(abs(x) for x in ray) for ray in rays))
        if checked != determinant:
            raise ArithmeticError('artifact determinant mismatch')
        simplices.append((rays, determinant))
    return simplices, hashlib.sha256(path.read_bytes()).hexdigest(), dict(exact.counts)


def main():
    started = time.process_time()
    output = dict(complete=False, model_forward_inputs=0,
        classification='conditional enclosure under documented IEEE754/NumPy elementary-operation assumptions',
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__)
    def save():
        output['cpu_seconds'] = time.process_time()-started
        output['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (HERE/'enclosed_h2_t4_results.json').write_text(json.dumps(output, indent=2)+'\n')
    try:
        with np.errstate(over='raise', divide='raise', invalid='raise', under='ignore'):
            output['arithmetic_controls'] = arithmetic_controls()
            simplices, artifact_hash, exact_work = load_exact_simplices()
            output['exact_artifact_sha256'] = artifact_hash
            output['artifact_determinant_recheck_work'] = exact_work
            clock = time.process_time()
            mass = EnclosedMass(order=20, seconds=max(1., 105-(time.process_time()-started)))
            output['rule_setup_cpu'] = time.process_time()-clock
            output['root_setup_cpu'] = mass.root_setup_cpu
            output['rational_moment_check_cpu'] = mass.moment_check_cpu
            output['rational_moment_checks'] = mass.moment_checks
            output['tensor_jacobian_interval'] = mass.jacobian_mass
            output['pi_interval'] = mass.pi
            output['setup_counts'] = mass.setup_counts
            certificate = dict(order=mass.order,
                roots=[[[str(v.numerator), str(v.denominator)] for v in pair] for pair in mass.root_intervals],
                weights=[[[str(v.numerator), str(v.denominator)] for v in pair] for pair in mass.weight_intervals],
                pi=[[str(v.numerator), str(v.denominator)] for v in mass.pi_rational])
            (HERE/'rule20_certificate.json').write_text(json.dumps(certificate, separators=(',', ':'))+'\n')
            identity = tuple(tuple(int(i == j) for j in range(4)) for i in range(4))
            orthant_value = mass.mass(identity, F(1))
            orthant_total = mass.total([orthant_value])
            if not F.from_float(orthant_total['probability_interval'][0]) <= F(1,16) <= F.from_float(orthant_total['probability_interval'][1]):
                raise ArithmeticError('known orthant probability not enclosed')
            output['orthant_control'] = orthant_total
            before_full = dict(mass.stats)
            clock = time.process_time()
            values = [mass.mass(rays, determinant) for rays, determinant in simplices]
            output['accuracy'] = mass.total(values)
            output['integration_cpu'] = time.process_time()-clock
            output['integration_counts'] = {key: value-before_full.get(key, 0) for key, value in mass.stats.items()}
            output['complete_counts'] = dict(mass.stats)
            output['simplices'] = len(simplices)
            output['per_simplex_intervals'] = [dict(quadrature=list(value[0]), truncation_upper=value[1]) for value in values]
            output['legacy_gl20_value'] = .3077307430239267
            output['legacy_inside_quadrature_interval'] = output['accuracy']['quadrature_interval'][0] <= output['legacy_gl20_value'] <= output['accuracy']['quadrature_interval'][1]
            output['complete'] = True
    except (ArithmeticError, ValueError, TimeoutError, AssertionError, FloatingPointError) as error:
        output['failure'] = repr(error)
    save()
    print(json.dumps({key:value for key,value in output.items() if key != 'per_simplex_intervals'}))

if __name__ == '__main__':
    main()
