"""Analytic inverse-sixth norm integral; no model calls or quadrature nodes.

The returned optional series bound concerns truncation alone, not floating-point
arithmetic. Non-series rows have NaN bounds: closed forms lack a rounding bound.
"""
import numpy as np


def inverse_six_segment(a, b, diagnostics=False, series_terms=28):
    """Integral_0^1 ||(1-u)*a+u*b||^-6 du, broadcasting endpoint batches.

    Endpoints must be nonzero and share an orthant (zero entries allowed).
    Shape (..., dimension) -> (...). All arithmetic uses float64.
    """
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if a.ndim < 1 or a.shape[-1] == 0:
        raise ValueError('endpoints need a final coordinate dimension')
    if series_terms < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite endpoints and positive series_terms required')
    if np.any(((a < 0) & (b > 0)) | ((a > 0) & (b < 0))):
        raise ValueError('endpoints must share an orthant')
    scale = np.maximum(np.max(np.abs(a), axis=-1), np.max(np.abs(b), axis=-1))
    if np.any(scale == 0):
        raise ValueError('zero endpoint')
    aa, bb = a / scale[..., None], b / scale[..., None]
    r, s = np.linalg.norm(aa, axis=-1), np.linalg.norm(bb, axis=-1)
    if np.any(r == 0) or np.any(s == 0):
        raise ValueError('zero endpoint')
    cosine = np.sum(aa * bb, axis=-1) / (r * s)
    # Cauchy roundoff can put a collinear pair a few ulps above one.
    cosine = np.clip(cosine, 0., 1.)
    z = (1. - cosine) * (1. + cosine)
    near = z <= .25
    coef_a = np.empty_like(z)
    remainder = np.full_like(z, np.nan)
    zz = z[near]
    term = np.full_like(zz, .2)
    total = term.copy()
    for n in range(series_terms - 1):
        term *= ((n + .5) / (n + 1)) * zz * ((2*n + 5) / (2*n + 7))
        total += term
    n = series_terms - 1
    next_term = term * ((n + .5)/(n + 1)) * zz * ((2*n + 5)/(2*n + 7))
    coef_a[near] = total
    remainder[near] = next_term / (1. - zz)
    far_z = z[~near]
    sine = np.sqrt(far_z)
    theta = np.arctan2(sine, cosine[~near])
    coef_a[~near] = (3*theta - sine*cosine[~near]*(3 + 2*far_z)) / (8*sine**5)
    coef_b = .25 - cosine*coef_a
    coef_c = (1 - 2*z/3)*coef_a - cosine/6
    rr, ss = 1/r, 1/s
    u = rr**4 + ss**4
    v = 4*(rr**3*ss + rr*ss**3)
    w = 6*rr**2*ss**2
    prefactor = rr*ss*scale**-6
    result = prefactor*(u*coef_a + v*coef_b + w*coef_c)
    if not diagnostics:
        return result
    # Absolute sum propagation avoids cancellation in the derivative w.r.t. A.
    truncation_bound = prefactor * (u + v*cosine + w*np.abs(1-2*z/3)) * remainder
    stats = {
        'endpoint_pairs': int(z.size),
        'coordinate_dimension': a.shape[-1],
        'norms': int(2*z.size),
        'dot_products': int(z.size),
        'series_pairs': int(np.count_nonzero(near)),
        'closed_form_pairs': int(np.count_nonzero(~near)),
        'series_terms_total': int(np.count_nonzero(near)*series_terms),
        'series_next_terms_for_bound': int(np.count_nonzero(near)),
        'model_forward_passes': 0,
        'quadrature_nodes': 0,
        'series_truncation_bound': truncation_bound,
        'bound_excludes_floating_arithmetic': True,
    }
    return result, stats
