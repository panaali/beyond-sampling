"""T72 positive T4 cubature enclosure under documented IEEE754 assumptions.

Exact rational root/weight/pi setup; every binary64 elementary operation is
outward rounded. No BLAS dot, floating determinant, or opaque sum reduction.
"""
from collections import defaultdict
from fractions import Fraction as F
import math
import time
import numpy as np
from numpy.polynomial.legendre import leggauss


def radd(a, b):
    return a[0]+b[0], a[1]+b[1]


def rsub(a, b):
    return a[0]-b[1], a[1]-b[0]


def rmul(a, b):
    terms = [x*y for x in a for y in b]
    return min(terms), max(terms)


def rdiv(a, b):
    if b[0] <= 0 <= b[1]:
        raise ArithmeticError('rational interval denominator contains zero')
    terms = [x/y for x in a for y in b]
    return min(terms), max(terms)


def rsquare(a):
    return (F(0) if a[0] <= 0 <= a[1] else min(a[0]**2, a[1]**2), max(a[0]**2, a[1]**2))


def exact_legendre(n, x, stats):
    stats['exact_legendre_evaluations'] += 1
    p, q = F(1), x
    if n == 0:
        return p
    for k in range(2, n+1):
        p, q = q, ((2*k-1)*x*q-(k-1)*p)/k
        stats['exact_legendre_recurrence_steps'] += 1
    return q


def interval_legendre(n, x, stats):
    p, q = (F(1), F(1)), x
    if n == 0:
        return p
    for k in range(2, n+1):
        qnext = rdiv(rsub(rmul((F(2*k-1),)*2, rmul(x, q)), rmul((F(k-1),)*2, p)), (F(k),)*2)
        p, q = q, qnext
        stats['rational_interval_legendre_steps'] += 1
    return q


def rational_rule(n, stats, deadline, bits=96):
    """Each of n disjoint intervals contains a different root by exact signs."""
    guesses, _ = leggauss(n)
    stats['uncertified_seed_rule_generations'] += 1
    roots = []
    for guess in guesses:
        center = F.from_float(float(guess))
        radius = F(1, 2**40)
        for attempt in range(20):
            lo, hi = center-radius, center+radius
            flo = exact_legendre(n, lo, stats)
            fhi = exact_legendre(n, hi, stats)
            if flo*fhi < 0:
                break
            radius *= 2
        else:
            raise ArithmeticError('failed exact root bracket')
        while hi-lo > F(1, 2**bits):
            if time.process_time() > deadline:
                raise TimeoutError('root isolation deadline')
            mid = (lo+hi)/2
            fm = exact_legendre(n, mid, stats)
            stats['root_bisections'] += 1
            if fm == 0:
                lo = hi = mid
                break
            if (fm > 0) == (flo > 0):
                lo, flo = mid, fm
            else:
                hi, fhi = mid, fm
        roots.append((lo, hi))
    if len(roots) != n or roots[0][0] <= -1 or roots[-1][1] >= 1:
        raise ArithmeticError('root range/count certificate failed')
    if any(left[1] >= right[0] for left, right in zip(roots, roots[1:])):
        raise ArithmeticError('root brackets overlap')
    weights = []
    for root in roots:
        p = interval_legendre(n-1, root, stats)
        square = rsquare(p)
        if square[0] <= 0:
            raise ArithmeticError('weight formula denominator not positive')
        weights.append(rdiv(rmul((F(2),)*2, rsub((F(1),)*2, rsquare(root))),
                            rmul((F(n*n),)*2, square)))
    stats['certified_root_intervals'] += n
    return roots, weights


def machin_pi(terms=40):
    def atan_inverse(a):
        partial = sum((F((-1)**k, (2*k+1)*a**(2*k+1)) for k in range(terms)), F(0))
        endpoint = partial+F((-1)**terms, (2*terms+1)*a**(2*terms+1))
        return min(partial, endpoint), max(partial, endpoint)
    return rsub(rmul((F(16),)*2, atan_inverse(5)), rmul((F(4),)*2, atan_inverse(239)))


def fraction_bounds(value):
    x = float(value)
    if not math.isfinite(x):
        raise ArithmeticError('nonfinite rational conversion')
    represented = F.from_float(x)
    lo = math.nextafter(x, -math.inf) if represented > value else x
    hi = math.nextafter(x, math.inf) if represented < value else x
    if not F.from_float(lo) <= value <= F.from_float(hi):
        raise ArithmeticError('rational conversion containment failed')
    return lo, hi


def interval_bounds(interval):
    return fraction_bounds(interval[0])[0], fraction_bounds(interval[1])[1]


class PositiveIntervals:
    """Intervals known nonnegative; complements additionally require [0,1]."""
    def __init__(self, stats):
        self.stats = stats

    def rounded(self, lower, upper, operation):
        lower, upper = np.broadcast_arrays(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64))
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ArithmeticError('nonfinite interval arithmetic')
        lo = np.maximum(0., np.nextafter(lower, -np.inf))
        hi = np.nextafter(upper, np.inf)
        if np.any(lo > hi) or np.any(hi < 0):
            raise ArithmeticError('invalid positive interval')
        self.stats[operation+'_scalar_operations'] += 2*lo.size
        self.stats['nextafter_scalar_operations'] += 2*lo.size
        return lo, hi

    def add(self, a, b):
        return self.rounded(np.add(a[0], b[0]), np.add(a[1], b[1]), 'add')

    def mul(self, a, b):
        return self.rounded(np.multiply(a[0], b[0]), np.multiply(a[1], b[1]), 'multiply')

    def div(self, a, b):
        if np.any(np.asarray(b[0]) <= 0):
            raise ArithmeticError('reciprocal denominator not positive')
        return self.rounded(np.divide(a[0], b[1]), np.divide(a[1], b[0]), 'divide')

    def complement(self, a):
        if np.any(np.asarray(a[0]) < 0) or np.any(np.asarray(a[1]) > 1):
            raise ArithmeticError('complement input outside [0,1]')
        return self.rounded(np.subtract(1., a[1]), np.subtract(1., a[0]), 'subtract')

    def sum(self, a):
        lo, hi = np.asarray(a[0]).ravel(), np.asarray(a[1]).ravel()
        if len(lo) == 0:
            return np.array(0.), np.array(0.)
        size = 1 << (len(lo)-1).bit_length()
        lo, hi = np.pad(lo, (0, size-len(lo))), np.pad(hi, (0, size-len(hi)))
        self.stats['pairwise_reduction_padding_entries'] += size-len(np.asarray(a[0]).ravel())
        while len(lo) > 1:
            lo, hi = self.add((lo[::2], hi[::2]), (lo[1::2], hi[1::2]))
        return lo[0], hi[0]


def check_moments(roots, weights, stats, deadline):
    powers = [(F(1), F(1)) for _ in roots]
    maximum_width = 0.
    for degree in range(2*len(roots)):
        if time.process_time() > deadline:
            raise TimeoutError('rational moment-check deadline')
        total = (F(0), F(0))
        for i, (root, weight) in enumerate(zip(roots, weights)):
            total = radd(total, rmul(weight, powers[i]))
            powers[i] = rmul(powers[i], root)
            stats['rational_moment_terms'] += 1
        truth = F(0) if degree % 2 else F(2, degree+1)
        if not total[0] <= truth <= total[1]:
            raise ArithmeticError(f'GL moment {degree} not enclosed')
        maximum_width = max(maximum_width, float(total[1]-total[0]))
    return dict(polynomial_degrees_checked=2*len(roots), maximum_rational_interval_width=maximum_width)


class EnclosedMass:
    def __init__(self, order=20, seconds=105, verify_moments=True):
        setup_started = time.process_time()
        self.order = order
        self.deadline = time.process_time()+seconds
        self.stats = defaultdict(int)
        self.ops = PositiveIntervals(self.stats)
        clock = time.process_time()
        roots, weights = rational_rule(order, self.stats, self.deadline)
        self.root_intervals = roots
        self.weight_intervals = weights
        self.pi_rational = machin_pi()
        self.pi = interval_bounds(self.pi_rational)
        self.root_setup_cpu = time.process_time()-clock
        clock = time.process_time()
        self.moment_checks = check_moments(roots, weights, self.stats, self.deadline) if verify_moments else None
        self.moment_check_cpu = time.process_time()-clock
        # Mapping to [0,1] is done rationally before outward float conversion.
        nodes = [interval_bounds(((lo+1)/2, (hi+1)/2)) for lo, hi in roots]
        quadrature_weights = [interval_bounds((lo/2, hi/2)) for lo, hi in weights]
        node_lower, node_upper = np.array(nodes).T
        weight_lower, weight_upper = np.array(quadrature_weights).T
        lows = np.meshgrid(node_lower, node_lower, node_lower, indexing='ij')
        highs = np.meshgrid(node_upper, node_upper, node_upper, indexing='ij')
        u, v, z = [(a.ravel(), b.ravel()) for a, b in zip(lows, highs)]
        weight_lows = np.meshgrid(weight_lower, weight_lower, weight_lower, indexing='ij')
        weight_highs = np.meshgrid(weight_upper, weight_upper, weight_upper, indexing='ij')
        wu, wv, wz = [(a.ravel(), b.ravel()) for a, b in zip(weight_lows, weight_highs)]
        one_u, one_v, one_z = self.ops.complement(u), self.ops.complement(v), self.ops.complement(z)
        uv = self.ops.mul(one_u, one_v)
        self.y = (u, self.ops.mul(one_u, v), self.ops.mul(uv, z), self.ops.mul(uv, one_z))
        jacobian = self.ops.mul(self.ops.mul(one_u, one_u), one_v)
        self.weight = self.ops.mul(self.ops.mul(self.ops.mul(wu, wv), wz), jacobian)
        jacobian_mass = self.ops.sum(self.weight)
        if not F.from_float(float(jacobian_mass[0])) <= F(1, 6) <= F.from_float(float(jacobian_mass[1])):
            raise ArithmeticError('tensor Jacobian integral not enclosed')
        self.jacobian_mass = tuple(float(x) for x in jacobian_mass)
        self.pi_squared = self.ops.mul(self.pi, self.pi)
        self.mass_denominator = self.ops.mul((2., 2.), self.pi_squared)
        tail = F(12*order+9, 3**(order-1))
        self.tail_per_det = self.ops.div(fraction_bounds(tail), self.ops.mul((6., 6.), self.pi_squared))
        self.setup_counts = dict(self.stats)
        self.setup_cpu = time.process_time()-setup_started

    def mass(self, integer_rays, determinant):
        """Return (quadrature enclosure, truncation-bound upper endpoint)."""
        if time.process_time() > self.deadline:
            raise TimeoutError('enclosed mass deadline')
        if len(integer_rays) != 4 or any(len(ray) != 4 for ray in integer_rays) or determinant <= 0:
            raise ValueError('positive full-dimensional R4 simplex required')
        for coordinate in zip(*integer_rays):
            if min(coordinate) < 0 < max(coordinate):
                raise ValueError('rays do not share an orthant')
        lengths = [sum(abs(x) for x in ray) for ray in integer_rays]
        squared_norm = (np.zeros(self.order**3), np.zeros(self.order**3))
        for axis in range(4):
            coordinate = (np.zeros(self.order**3), np.zeros(self.order**3))
            for j, ray in enumerate(integer_rays):
                ratio = F(abs(ray[axis]), lengths[j])
                self.stats['exact_ray_coordinate_conversions'] += 1
                coordinate = self.ops.add(coordinate, self.ops.mul(self.y[j], fraction_bounds(ratio)))
            squared_norm = self.ops.add(squared_norm, self.ops.mul(coordinate, coordinate))
        denominator = self.ops.mul(squared_norm, squared_norm)
        integral = self.ops.sum(self.ops.div(self.weight, denominator))
        determinant_interval = fraction_bounds(determinant)
        factor = self.ops.div(determinant_interval, self.mass_denominator)
        value = self.ops.mul(factor, integral)
        error = self.ops.mul(determinant_interval, self.tail_per_det)[1]
        self.stats['simplices_integrated'] += 1
        self.stats['quadrature_nodes'] += self.order**3
        return tuple(float(x) for x in value), float(error)

    def total(self, values):
        lower = np.array([v[0][0] for v in values])
        upper = np.array([v[0][1] for v in values])
        qlo, qhi = self.ops.sum((lower, upper))
        errors = np.array([v[1] for v in values])
        error = float(self.ops.sum((errors, errors))[1])
        lo = math.nextafter(float(qlo)-error, -math.inf)
        hi = math.nextafter(float(qhi)+error, math.inf)
        self.stats['final_signed_add_subtract_operations'] += 2
        self.stats['nextafter_scalar_operations'] += 2
        return dict(quadrature_interval=[float(qlo), float(qhi)],
                    truncation_error_upper=error, probability_interval=[lo, hi],
                    quadrature_interval_width=float(qhi-qlo), total_width=hi-lo)
