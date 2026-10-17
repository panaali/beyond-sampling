"""Guarded stored-binary64 determinant enclosure, not a native cost claim."""
from collections import Counter
from fractions import Fraction
import math
import sys

LOW, HIGH, FACTOR = 2.0**-100, 2.0**100, 2.0**-50


class Ledger:
    def __init__(self):
        self.counts = Counter()
        self.max_numerator_bits = 0
        self.max_denominator_bits = 0

    def record_exact(self, x):
        self.max_numerator_bits = max(self.max_numerator_bits, abs(x.numerator).bit_length())
        self.max_denominator_bits = max(self.max_denominator_bits, x.denominator.bit_length())
        return x

    def operation(self, name, x, y, exact=False):
        self.counts[('exact_' if exact else 'float_')+name] += 1
        if name == 'mul':
            z = x*y
        elif name == 'add':
            z = x+y
        else:
            assert name == 'sub'
            z = x-y
        return self.record_exact(z) if exact else z

    def compare(self, name, x, y):
        self.counts['comparison'] += 1
        return x == y if name == 'eq' else x <= y

    def neighbor(self, x, direction):
        self.counts['nextafter'] += 1
        return math.nextafter(x, direction)


def tree(a, ledger, positive=False, exact=False):
    """The frozen separate-operation tree: nine products and five sums."""
    x0, x1, x2, x3, x4, x5, x6, x7, x8 = a
    op = lambda name, x, y: ledger.operation(name, x, y, exact)
    minor = 'add' if positive else 'sub'
    p0 = op('mul', x4, x8); p1 = op('mul', x5, x7)
    m0 = op(minor, p0, p1); q0 = op('mul', x0, m0)
    p2 = op('mul', x3, x8); p3 = op('mul', x5, x6)
    m1 = op(minor, p2, p3); q1 = op('mul', x1, m1)
    p4 = op('mul', x3, x7); p5 = op('mul', x4, x6)
    m2 = op(minor, p4, p5); q2 = op('mul', x2, m2)
    r = op(minor, q0, q1)
    return op('add', r, q2)


def exact_fallback(a, ledger):
    rationals = []
    for x in a:
        ledger.counts['float_to_exact'] += 1
        rationals.append(ledger.record_exact(Fraction.from_float(x)))
    determinant = tree(rationals, ledger, exact=True)
    ledger.counts['exact_to_float'] += 1
    try:
        point = float(determinant)
    except OverflowError:
        ledger.counts['exact_comparison'] += 1
        point = math.inf if determinant > 0 else -math.inf
    ledger.counts['isfinite'] += 1
    if not math.isfinite(point):
        ledger.counts['comparison'] += 1
        if point > 0:
            lo, hi = sys.float_info.max, math.inf
        else:
            lo, hi = -math.inf, -sys.float_info.max
    else:
        # Two directed neighbors also handle conversion underflow to signed zero.
        lo = ledger.neighbor(point, -math.inf)
        hi = ledger.neighbor(point, math.inf)
    return dict(status='exact_fallback', point=point, lower=lo, upper=hi,
                exact=[str(determinant.numerator), str(determinant.denominator)])


def enclose(matrix):
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    ledger = Ledger()
    a = []
    for row in matrix:
        for x in row:
            ledger.counts['to_float'] += 1
            a.append(float(x))
    finite = True
    for x in a:
        ledger.counts['isfinite'] += 1
        finite = math.isfinite(x) and finite
    if not finite:
        result = dict(status='unknown_nonfinite')
    else:
        absolute = []
        eligible = True
        for x in a:
            ledger.counts['abs'] += 1
            z = abs(x)
            absolute.append(z)
            ok = (ledger.compare('eq', z, 0.0) or
                  (ledger.compare('le', LOW, z) and ledger.compare('le', z, HIGH)))
            eligible = ok and eligible
        if not eligible:
            result = exact_fallback(a, ledger)
        else:
            point = tree(a, ledger)
            permanent = tree(absolute, ledger, positive=True)
            error = ledger.operation('mul', FACTOR, permanent)
            if ledger.compare('eq', error, 0.0):
                assert ledger.compare('eq', point, 0.0)
                lo = hi = 0.0
            else:
                lo = ledger.neighbor(ledger.operation('sub', point, error), -math.inf)
                hi = ledger.neighbor(ledger.operation('add', point, error), math.inf)
            result = dict(status='fast_enclosure', point=point, lower=lo, upper=hi,
                          permanent=permanent, error_allowance=error)
    result.update(counts=dict(ledger.counts),
                  fallback_max_numerator_bits=ledger.max_numerator_bits,
                  fallback_max_denominator_bits=ledger.max_denominator_bits)
    return result
