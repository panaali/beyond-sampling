"""Eliminate two simplex coordinates using weighted spherical moments.

The prototype prioritizes explicit diagnostics over batching. Fallback bounds
are quadrature-only in real arithmetic, not floating-point certificates.
"""
from math import exp, lgamma, log
import numpy as np
from moments import polygon_moments, area
from positive import rule


def _radial_bounds(vertices):
    center = np.sum(vertices, axis=0)
    center /= np.linalg.norm(center)
    qlo = float(np.min(vertices @ center))**2
    qhi = float(np.max(np.sum(vertices**2, axis=1)))
    return min(qlo, qhi), qhi


def _positive_original(vertices, exponent, tolerance, max_order):
    qlo, qhi = _radial_bounds(vertices)
    mid = (qlo+qhi)/2
    ratio = (qhi-qlo)/(qhi+qlo)
    power = exponent/2
    choice = None
    bound_trials = 0
    lgamma_calls = 0
    bound_exp_log_calls = 0
    for n in (4,8,12,16,24,32,48,64,96,128):
        if n > max_order:
            continue
        # Duffy degree 2*K+1 <= 2*n-1, hence K=n-1 and N=n.
        rho = ratio*(power+n)/(n+1)
        if rho >= 1:
            continue
        bound_trials += 1
        if ratio != 0:
            lgamma_calls += 3
            bound_exp_log_calls += 1
        error = 0. if ratio == 0 else mid**-power * exp(
            lgamma(power+n)-lgamma(power)-lgamma(n+1)+n*log(ratio))/(1-rho)
        if error <= tolerance:
            choice = n,error
            break
    if choice is None:
        raise ArithmeticError('original-triangle quadrature bound unavailable at max_order')
    n,error = choice
    before = rule.cache_info().misses
    bary, weights = rule(n)
    generations = rule.cache_info().misses-before
    points = vertices.T @ bary
    q = np.sum(points*points,axis=0)
    value = float(weights @ q**(-power))
    return value, {'quadrature_nodes': n*n, 'quadrature_error_bound':error,
                   'quadrature_order':n, 'legendre_rule_generations':generations,
                   'legendre_rule_generation_nodes':n*generations,
                   'qmin':qlo, 'qmax':qhi, 'binomial_ratio':ratio,
                   'bound_trials':bound_trials, 'bound_lgamma_calls':lgamma_calls,
                   'bound_exp_calls':bound_exp_log_calls, 'bound_log_calls':bound_exp_log_calls}


def triangle_inverse_norm(vertices, exponent=None, absolute_tolerance=1e-11,
                          min_singular_ratio=1e-5, max_order=128,
                          known_plane_normal=None):
    """Integrate ||y0*A+y1*B+y2*C||^-exponent dy over the unit simplex.

    vertices shape(3,ambient_dimension), in one closed orthant, each nonzero.
    exponent defaults to ambient_dimension, and must be an integer >=3.
    Returns (value,metadata), including method and all fallback node counts.
    Optional known_plane_normal must satisfy vertices@normal=1. For signed-L1
    simplex endpoints it is the orthant sign vector; projection then replaces
    an ill-conditioned plane solve. Its residual is checked, not certified.
    """
    vertices = np.asarray(vertices,dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] != 3 or vertices.shape[1] < 3:
        raise ValueError('need three endpoint rows in ambient dimension >=3')
    if not np.isfinite(vertices).all() or np.any(np.linalg.norm(vertices,axis=1)==0):
        raise ValueError('need finite nonzero endpoints')
    if np.any((np.min(vertices,axis=0)<0)&(np.max(vertices,axis=0)>0)):
        raise ValueError('endpoints must share an orthant')
    exponent = vertices.shape[1] if exponent is None else exponent
    if int(exponent) != exponent or exponent < 3 or absolute_tolerance <= 0:
        raise ValueError('integer exponent >=3 and positive tolerance required')
    exponent = int(exponent)
    if known_plane_normal is not None:
        normal=np.asarray(known_plane_normal,dtype=float)
        if normal.shape != (vertices.shape[1],) or not np.isfinite(normal).all():
            raise ValueError('known plane normal has wrong shape or nonfinite entries')
        plane_residual=float(np.max(np.abs(vertices@normal-1)))
        if plane_residual > 1e-10:
            raise ValueError('known plane normal does not put endpoints at level one')
    scale = float(np.max(np.abs(vertices)))
    vv = vertices/scale
    factor = scale**-exponent
    tolerance = absolute_tolerance/factor
    stats = {'qr_factorizations':1, 'svd_condition_estimates':1, 'linear_solves':0,
             'determinants':0, 'spherical_moment_calls':0, 'spherical_area_calls':0,
             'great_circle_edges':0, 'great_circle_arctan2_calls':0,
             'solid_angle_arctan2_calls':0, 'moment_complex_exp_calls':0,
             'moment_sinc_calls':0,
             'positive_quadrature_fallbacks':0, 'quadrature_nodes':0,
             'quadrature_error_bound':0., 'legendre_rule_generations':0,
             'legendre_rule_generation_nodes':0, 'model_forward_passes':0,
             'roundoff_certified':False, 'ambient_dimension':vertices.shape[1],
             'exponent':exponent, 'known_normal_projections':0}
    qq, rr = np.linalg.qr(vv.T,mode='reduced')
    singular = np.linalg.svd(rr,compute_uv=False)
    condition_ratio = float(singular[-1]/singular[0])
    stats['singular_ratio'] = condition_ratio
    reason = None
    value = None
    if condition_ratio < min_singular_ratio:
        reason = 'small singular-value ratio'
    else:
        if known_plane_normal is None:
            stats['linear_solves'] += 1
            cc = np.linalg.solve(rr.T,np.ones(3))
        else:
            stats['known_normal_projections'] += 1
            stats['known_plane_residual'] = plane_residual
            cc = qq.T @ (scale*normal)
        cc_norm = np.linalg.norm(cc)
        direction = cc/cc_norm
        shift = direction-np.array([1.,0.,0.])
        denom = shift@shift
        rotation = np.eye(3) if denom < 1e-30 else np.eye(3)-2*np.outer(shift,shift)/denom
        rays = rr.T @ rotation
        rays /= np.linalg.norm(rays,axis=1)[:,None]
        stats['determinants'] += 1
        determinant = abs(float(np.linalg.det(rr)))
        stats['spherical_moment_calls'] += 1
        stats['spherical_moment_degree'] = exponent-3
        stats['great_circle_edges'] += 3
        stats['great_circle_arctan2_calls'] += 3
        # polygon_moments computes one spherical area and one finite Fourier
        # integral per edge and degree. Counts exclude library internals.
        stats['determinants'] += 1
        stats['solid_angle_arctan2_calls'] += 1
        stats['moment_complex_exp_calls'] += 3*(exponent-3)*(exponent-2)//2
        stats['moment_sinc_calls'] = stats['moment_complex_exp_calls']
        try:
            moment = polygon_moments(rays,exponent-3)[-1]
            value = cc_norm**(exponent-3)*moment/determinant
            qlo,qhi = _radial_bounds(vv)
            lower,upper = .5*qhi**(-exponent/2), .5*qlo**(-exponent/2)
            # Additional spherical bound catches large errors masked by det/c.
            stats['spherical_area_calls'] += 1
            stats['determinants'] += 1
            stats['solid_angle_arctan2_calls'] += 1
            solid_angle = area(rays)
            min_coordinate = np.min(rays[:,0])
            max_coordinate = min(1., np.max(rays[:,0]) / np.min(rays @ (
                np.sum(rays,axis=0)/np.linalg.norm(np.sum(rays,axis=0)))))
            moment_lo = solid_angle*min_coordinate**(exponent-3)
            moment_hi = solid_angle*max_coordinate**(exponent-3)
            if (not np.isfinite(value) or value < lower*(1-1e-11) or value > upper*(1+1e-11)
                    or moment < moment_lo*(1-1e-11) or moment > moment_hi*(1+1e-11)):
                reason = 'analytic moment violates positive geometric bounds'
            stats.update(plane_normal_norm=float(cc_norm), normalized_determinant=determinant,
                         qmin=qlo,qmax=qhi)
        except (ArithmeticError,ValueError) as exc:
            reason = 'analytic moment failed: '+str(exc)
    if reason:
        value, meta = _positive_original(vv,exponent,tolerance,max_order)
        stats.update(meta)
        stats['quadrature_error_bound'] *= factor
        stats['positive_quadrature_fallbacks'] = 1
        stats['fallback_reason'] = reason
        stats['method'] = 'positive_original_triangle_quadrature'
    else:
        stats['method'] = 'analytic_weighted_spherical_triangle'
    return factor*value, stats
