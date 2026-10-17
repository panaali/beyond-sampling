"""Synthetic geometry only: validate gate-supported degree-one projection.

No trained weights, model inputs, sampling, or estimator performance claims.
All directions in these checks lie in the first two coordinates of S^(n-1).
"""
import math


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def estimate(n, degrees):
    c = math.gamma(n / 2) / (2 * math.sqrt(math.pi) * math.gamma((n + 1) / 2))
    directions = [(math.cos(math.radians(t)), math.sin(math.radians(t))) for t in degrees]
    previous, a, b = directions[0], 1., [0., 0.]
    for v in directions[1:]:
        theta = math.acos(max(-1., min(1., dot(previous, v))))
        length = math.pi - theta
        p = length / (2 * math.pi)
        m = [c * (x + y) / 2 for x, y in zip(previous, v)]
        rhs0 = p * a + dot(m, b)
        rhs = [a * x + p * y / n for x, y in zip(m, b)]
        if theta > 1e-10 and math.pi - theta > 1e-10:
            w = [(x + y) / (2 * math.cos(theta / 2)) for x, y in zip(previous, v)]
            z = [(x - y) / (2 * math.sin(theta / 2)) for x, y in zip(previous, v)]
            wb, zb = dot(w, b), dot(z, b)
            gamma = math.sin(length) / (2 * math.pi * n)
            rhs = [x + gamma * (y * wb - z0 * zb) for x, y, z0 in zip(rhs, w, z)]
        a = (rhs0 - 2 * n * c * dot(v, rhs)) / (.5 - 2 * n * c * c)
        b = [2 * n * (x - c * y * a) for x, y in zip(rhs, v)]
        previous = v
    return .5 * a + c * dot(previous, b)


def main():
    for n in (3, 10):
        c = math.gamma(n / 2) / (2 * math.sqrt(math.pi) * math.gamma((n + 1) / 2))
        print(f'n={n} global-D2 duplicate={.25+n*c*c:.9f} opposite={.25-n*c*c:.9f}')
        for angles in ((0, 0), (0, 90), (0, 180), (0, 45, 90), (0, 90, 45), (0, 30, 60, 90), (0, 90, 0)):
            # These normals have angular span <= pi, so the intersection is one wedge.
            exact = (180 - (max(angles) - min(angles))) / 360
            value = estimate(n, angles)
            if len(angles) == 2:
                assert abs(value - exact) < 1e-12
            print(f'{angles}: supported-D1={value:.9f} exact={exact:.9f} bias={value-exact:+.9f}')
        # Rotating the entire instance cannot change its spherical measure or projection.
        assert abs(estimate(n, (17, 62, 107)) - estimate(n, (0, 45, 90))) < 1e-12


if __name__ == '__main__':
    main()
