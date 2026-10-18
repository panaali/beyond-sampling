"""Exact finite rational identities; no Gaussian numerical evaluator."""
from fractions import Fraction


def rational(x):
    if type(x) not in (int, Fraction):
        raise TypeError('only exact int/Fraction inputs; bool and floats rejected')
    return Fraction(x)


def point(a, b):
    a, b = rational(a), rational(b)
    surrogate = b if a < 0 else Fraction(0)
    residual = abs(b) if ((a < 0 and b < 0) or (a >= 0 and b > 0)) else Fraction(0)
    return {'surrogate': surrogate, 'residual': residual}


def conditional(alpha, beta, c, d):
    alpha, beta, c, d = map(rational, (alpha, beta, c, d))
    if alpha * beta >= 0:
        raise ValueError('opposite nonzero fresh-input slopes required')
    u, v = -c / alpha, -d / beta
    delta = abs(u - v)
    # Integral of |beta| |x-v| over the mismatch interval, before density.
    area = abs(beta) * delta * delta / 2
    return {'u': u, 'v': v, 'lower': min(u, v), 'upper': max(u, v),
            'delta': delta, 'area': area,
            'anchor_endpoint_residual': max(beta * (u - v), Fraction(0))}


def interval_point(alpha, beta, c, d, x):
    alpha, beta, c, d, x = map(rational, (alpha, beta, c, d, x))
    q = conditional(alpha, beta, c, d)
    if q['lower'] < x < q['upper']:
        r = abs(beta) * abs(x - q['v'])
    elif x == q['u']:
        r = q['anchor_endpoint_residual']
    else:
        r = Fraction(0)
    return q, r


def scale(alpha, beta, c, d, di, dj, margin):
    alpha, beta, c, d, di, dj, margin = map(rational, (alpha, beta, c, d, di, dj, margin))
    if di <= 0 or dj <= 0 or margin < 0:
        raise ValueError('positive coordinate scales and nonnegative margin envelope required')
    original = conditional(alpha, beta, c, d)
    transformed = conditional(di * alpha, dj * beta, di * c, dj * d)
    return {'original': original, 'transformed': transformed,
            'weighted_area': margin * original['area'],
            'transformed_weighted_area': (margin / dj) * transformed['area']}
