"""Analytic moments of coordinate 0 over a convex spherical polygon on S².

No network evaluations or quadrature are used here. This floating-point primitive
is not a certificate; thin polygons can exhibit severe relative cancellation.
"""
from math import comb, pi
import numpy as np


def unit(x):
    x = np.asarray(x, dtype=float)
    scale = np.max(np.abs(x))
    if not scale:
        raise ValueError("zero ray")
    x = x / scale
    return x / np.linalg.norm(x)


def area(vertices):
    vertices = np.asarray(vertices, dtype=float)
    a = vertices[0]
    return sum(2 * np.arctan2(abs(np.linalg.det(np.stack((a, b, c)))),
                             1 + a @ b + b @ c + c @ a)
               for b, c in zip(vertices[1:-1], vertices[2:]))


def arc_power(a, b, theta, power):
    """Integral of (a cos(s) + b sin(s))**power from 0 to theta.

    A finite Fourier expansion uses sinc to avoid subtracting nearby exponentials.
    It can still cancel between Fourier modes; callers must preserve diagnostics.
    """
    c = (a - 1j * b) / 2
    value = 0j
    for m in range(power + 1):
        frequency = 2 * m - power
        integral = (theta * np.exp(0.5j * frequency * theta)
                    * np.sinc(frequency * theta / (2 * pi)))
        value += comb(power, m) * c**m * c.conjugate()**(power-m) * integral
    return float(value.real)


def polygon_moments(vertices, degree):
    """Return ∫ v[0]**p dΩ, p=0..degree; vertices ordered either way.

    Requires a convex minor-arc polygon in v0>=0, contained in a hemisphere.
    The center sum must be interior; no consecutive equal or antipodal rays.
    """
    vertices = np.array([unit(v) for v in vertices])
    if len(vertices) < 3 or degree < 0:
        raise ValueError("need a polygon and a nonnegative degree")
    if np.min(vertices[:, 0]) < -1e-14:
        raise ValueError("polygon must lie in the positive coordinate hemisphere")
    center = vertices.sum(axis=0)
    boundary = np.zeros(degree)
    for u, v in zip(vertices, np.roll(vertices, -1, axis=0)):
        cross = np.cross(u, v)
        length = np.linalg.norm(cross)
        if not length:
            raise ValueError("coincident or antipodal consecutive rays")
        inward = cross / length
        side = inward @ center
        if not side:
            raise ValueError("degenerate polygon")
        inward *= np.sign(side)
        outward0 = -inward[0]
        theta = np.arctan2(length, u @ v)
        tangent = np.cross(cross / length, u)
        for p in range(degree):
            boundary[p] += outward0 * arc_power(u[0], tangent[0], theta, p)
    result = np.zeros(degree + 1)
    result[0] = area(vertices)
    for p in range(1, degree + 1):
        previous = (p-1) * result[p-2] if p > 1 else 0.
        result[p] = (previous - boundary[p-1]) / (p+1)
    return result


def tail_mass(vertices, prefix_dimension):
    if prefix_dimension < 1:
        raise ValueError("k=0 needs a separate circle calculation")
    degree = prefix_dimension-1
    value = polygon_moments(vertices, degree)[-1]
    # Every point is normalize(sum_i y_i vertex_i), y_i>=0, sum_i y_i=1.
    # An interior separating direction gives a positive denominator bound.
    rays = np.array([unit(v) for v in vertices])
    center = unit(rays.sum(axis=0))
    denominator = np.min(rays @ center)
    if denominator <= 0:
        raise ArithmeticError("no simple positive moment bound; independent evaluation required")
    coordinate_bound = min(1., np.max(rays[:, 0])/denominator)
    upper = area(rays) * coordinate_bound**degree
    if value < 0 or value > upper*(1+1e-12):
        raise ArithmeticError("moment violates geometric bound; higher precision required")
    return prefix_dimension / (2*pi) * value


def tail_mass_with_fallback(vertices,prefix_dimension,stats=None,absolute_tolerance=1e-12):
    """Use analytic recurrence, then positive integration on explicit failure.

    Fallback error accounting covers quadrature only, not geometric/roundoff error.
    The old tail_mass remains available to expose the original failure cases.
    """
    try:
        return tail_mass(vertices,prefix_dimension)
    except ArithmeticError:
        from positive import polygon_tail_mass
        result,metadata=polygon_tail_mass(vertices,prefix_dimension,absolute_tolerance)
        if stats is not None:
            stats["positive_quadrature_fallbacks"]=stats.get("positive_quadrature_fallbacks",0)+1
            # Compatibility alias; neither key means high-precision arithmetic.
            stats["positive_fallbacks"]=stats.get("positive_fallbacks",0)+1
            stats["fallback_quadrature_nodes"]=stats.get("fallback_quadrature_nodes",0)+metadata["quadrature_nodes"]
            stats["fallback_quadrature_error_bound"]=stats.get("fallback_quadrature_error_bound",0.)+metadata["quadrature_error_bound"]
            stats["fallback_rule_generations"]=stats.get("fallback_rule_generations",0)+metadata.get("legendre_rule_generations",0)
            stats["fallback_rule_generation_nodes"]=stats.get("fallback_rule_generation_nodes",0)+metadata.get("legendre_rule_generation_nodes",0)
        return result
