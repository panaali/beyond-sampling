"""Frozen25 primitive controls; independent exact reference is permutation based."""
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time
from primitive import enclose

HERE = Path(__file__).resolve().parent


def diagonal(a, b, c):
    return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]


def near(k, scale=1.0):
    d = 2.0**-k
    return [[scale*x for x in row] for row in [[1., 1., 1.], [1., 1.+d, 1.], [1., 1., 1.+d]]]


def fixtures():
    yield 'zero', [[0.0]*3 for _ in range(3)]
    yield 'negative_zero', [[-0.0]*3 for _ in range(3)]
    yield 'identity', diagonal(1., 1., 1.)
    yield 'negative_diagonal', diagonal(-1., 2., 3.)
    yield 'cyclic', [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]
    yield 'dense_integer', [[1., 2., 3.], [0., 4., 5.], [1., 0., 6.]]
    yield 'duplicate_rows', [[1., 2., 3.], [1., 2., 3.], [4., 5., 6.]]
    yield 'ones', [[1.]*3 for _ in range(3)]
    for k in (50, 40, 26):
        yield 'near_'+str(k), near(k)
    yield 'large_diagonal', diagonal(2.**100, 2.**100, 2.**100)
    yield 'small_diagonal', diagonal(2.**-100, 2.**-100, 2.**-100)
    yield 'mixed_diagonal', diagonal(2.**-100, 1., 2.**100)
    yield 'large_cancel', near(50, 2.**99)
    yield 'small_cancel', near(50, 2.**-99)
    yield 'below_guard', diagonal(math.nextafter(2.**-100, 0.), 1., 1.)
    yield 'above_guard', diagonal(math.nextafter(2.**100, math.inf), 1., 1.)
    yield 'subnormal', diagonal(math.nextafter(0., 1.), 1., 1.)
    yield 'overflow', diagonal(1e308, 1e308, 1e308)
    yield 'underflow', diagonal(1e-200, 1e-200, 1e-200)
    for name, x in [('nan', math.nan), ('posinf', math.inf), ('neginf', -math.inf)]:
        yield name, diagonal(x, 1., 1.)
    yield 'signed_zero_identity', [[1. if i == j else -0. for j in range(3)] for i in range(3)]


def exact_reference(matrix):
    q = [[Fraction.from_float(x) for x in row] for row in matrix]
    result = Fraction()
    for p in itertools.permutations(range(3)):
        parity = sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3))
        term = q[0][p[0]]*q[1][p[1]]*q[2][p[2]]
        result += (-1 if parity % 2 else 1)*term
    return result


def main():
    assert not (HERE/'results.json').exists(), 'Preserve prior results'
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    start = time.process_time()
    assert (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.max_exp,
            sys.float_info.min_exp, sys.float_info.rounds) == (2, 53, 1024, -1021, 1)
    rows = []
    for name, matrix in fixtures():
        row = dict(name=name, input_hex=[[x.hex() for x in r] for r in matrix])
        got = enclose(matrix)
        row['result'] = {k: (v.hex() if isinstance(v, float) else v) for k, v in got.items()}
        if any(not math.isfinite(x) for r in matrix for x in r):
            assert got['status'] == 'unknown_nonfinite'
            assert not any(k.startswith('exact_') or k == 'float_to_exact' for k in got['counts'])
            row['unknown_correctly_retained'] = True
        else:
            q = exact_reference(matrix)
            lo, hi = got['lower'], got['upper']
            assert (lo == -math.inf or Fraction.from_float(lo) <= q)
            assert (hi == math.inf or q <= Fraction.from_float(hi))
            row.update(exact=[str(q.numerator), str(q.denominator)], enclosed=True,
                       interval_contains_zero=lo <= 0 <= hi,
                       zero_point_nonzero_truth=got['point'] == 0 and q != 0)
            if math.isfinite(got['point']):
                err = Fraction.from_float(got['point'])-q
                row['point_error_exact'] = [str(err.numerator), str(err.denominator)]
                row['relative_point_error'] = None if q == 0 else str(abs(err/q))
        rows.append(row)
    assert len(rows) == 25
    u = Fraction(1, 2**53)
    gap = 8*(1-u)**5*(1-5*u)-5
    assert gap > 0
    result = dict(complete=True, rows=rows, gamma_inequality_gap=[str(gap.numerator), str(gap.denominator)],
                  python_version=sys.version, binary64_round_to_nearest_environment=True,
                  source_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                 for name in ('primitive.py', 'controls.py', 'PROOF.md', 'PROTOCOL.md')},
                  cpu_seconds=time.process_time()-start, new_primitive_calls=25,
                  new_mass_model_MC_calls=0,
                  scope='Frozen primitive enclosures only; source counters are not installed native counts. No downstream mass accuracy, sampler or full cheap-mass claim.')
    (HERE/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=True, cpu_seconds=result['cpu_seconds'],
                         statuses={s: sum(r['result']['status'] == s for r in rows)
                                   for s in ('fast_enclosure', 'exact_fallback', 'unknown_nonfinite')},
                         zero_point_failures=[r['name'] for r in rows if r.get('zero_point_nonzero_truth')])) )


if __name__ == '__main__':
    main()
