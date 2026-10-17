"""T5 tetra moments with a single fixed tangent frame per signed-L1 cone.

Rows of rays are five endpoints, not the column convention of geometry.py.
The conditioning and cancellation guards are conservative heuristics in float.
"""
import numpy as np
from primitive import positive_fallback, radial_interval


def fixed_tangent_tetra(rays, nodes, absolute_tolerance=1e-11,
                       min_singular_ratio=1e-6, roundoff_screen=1e-10,
                       frame_builder=None, fallback=None):
    v = np.asarray(rays, float)
    u = np.asarray(nodes, float)
    if v.ndim != 3 or v.shape[1:] != (5, 5):
        raise ValueError('need (cones,5,5) endpoint rows')
    if not np.isfinite(v).all():
        raise ValueError('finite rays required')
    signs = np.sign(v.sum(1))
    if np.max(abs(np.einsum('nvd,nd->nv', v, signs)-1)) > 1e-10:
        raise ValueError('shared signed-normal affine plane required')
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError('open interval nodes required')
    nc, nn = len(v), len(u)
    tolerance = np.broadcast_to(np.asarray(absolute_tolerance, float), (nc, nn))
    edges = np.swapaxes(v[:, 2:] - v[:, 1:2], 1, 2)
    if frame_builder is None:
        frame, edge_coordinates = np.linalg.qr(edges, mode='reduced')
        singular = np.linalg.svd(edge_coordinates, compute_uv=False)
        frame_good = singular[:, -1] >= min_singular_ratio * singular[:, 0]
    else:
        frame, frame_good = frame_builder(edges)
    tangent = v @ frame
    residual = v[:, :2] - tangent[:, :2] @ np.swapaxes(frame, 1, 2)
    l = 1-u
    b = u[None, :, None]*residual[:, 0, None] + l[None, :, None]*residual[:, 1, None]
    delta2 = np.sum(b*b, axis=-1)
    endpoint_tangent = u[None, :, None, None]*tangent[:, 0, None, None] + l[None, :, None, None]*tangent[:, None, 1:]
    endpoints = u[None, :, None, None]*v[:, 0, None, None] + l[None, :, None, None]*v[:, None, 1:]
    edge = tangent[:, 2:] - tangent[:, 1:2]
    cross = np.stack([np.cross(edge[:, 1], edge[:, 2]),
                      np.cross(edge[:, 2], edge[:, 0]),
                      np.cross(edge[:, 0], edge[:, 1])], axis=1)
    determinant = np.sum(edge[:, 0]*cross[:, 0], axis=1)
    valid = frame_good & (determinant != 0)
    inverse = np.zeros_like(cross)
    inverse[valid] = cross[valid]/determinant[valid, None, None]
    transported = np.einsum('nij,nkj->nki', inverse, endpoint_tangent[:, :, 0])
    invnorm2 = 1/delta2 + np.sum(transported**2, axis=-1)/(l[None, :]**2*delta2) + np.sum(inverse**2, axis=(1, 2))[:, None]/l[None, :]**2
    rnorm2 = np.sum(endpoints**2, axis=(2, 3))
    ratio_lower = 1/np.sqrt(7*invnorm2*rnorm2)
    eligible = valid[:, None] & (ratio_lower >= min_singular_ratio)
    lengths = np.linalg.norm(endpoints, axis=-1)
    unit = endpoints/lengths[..., None]
    terms = np.zeros((nc, nn, 4))
    for i in range(4):
        j, k, m = [z for z in range(4) if z != i]
        normal = np.cross(tangent[:, k+1]-tangent[:, j+1], tangent[:, m+1]-tangent[:, j+1])
        offset = np.einsum('nkd,nd->nk', endpoint_tangent[:, :, j], normal)
        cof_norm = np.sqrt(offset**2 + delta2*np.sum(normal**2, axis=1)[:, None])
        numerator = l[None, :]**2*cof_norm/(lengths[:, :, j]*lengths[:, :, k]*lengths[:, :, m])
        denominator = 1 + np.sum(unit[:, :, j]*unit[:, :, k], axis=-1) + np.sum(unit[:, :, k]*unit[:, :, m], axis=-1) + np.sum(unit[:, :, m]*unit[:, :, j], axis=-1)
        angle = 2*np.arctan2(numerator, denominator)
        terms[:, :, i] = np.divide(np.sign(determinant)[:, None]*(-1.)**i*offset*angle, cof_norm, out=np.zeros_like(offset), where=cof_norm != 0)
    numerator = terms.sum(-1)
    denom = 3*delta2*l[None, :]**3*abs(determinant)[:, None]
    value = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom != 0)
    screen = 32*np.finfo(float).eps*np.sum(abs(terms), axis=-1)/np.maximum(abs(numerator), np.finfo(float).tiny)
    lower, upper = radial_interval(endpoints)
    accept = eligible & np.isfinite(value) & (value >= upper**-2.5/6*(1-1e-10)) & (value <= lower**-2.5/6*(1+1e-10)) & (screen <= roundoff_screen)
    error = np.zeros_like(value)
    nodes_used = trials = 0
    for i, j in np.argwhere(~accept):
        value[i, j], error[i, j], count, attempt = (positive_fallback if fallback is None else fallback)(endpoints[i, j], float(tolerance[i, j]))
        nodes_used += count
        trials += attempt
    stats = dict(tetrahedra=nc*nn, analytic_tetrahedra=int(accept.sum()), positive_fallbacks=int((~accept).sum()),
                 fallback_segment_nodes=nodes_used, fallback_bound_trials=trials,
                 qr_factorizations=nc if frame_builder is None else 0,
                 svd_condition_estimates=nc if frame_builder is None else 0,
                 tangent_cross_products=7*nc, facet_atan2_calls=4*nc*nn,
                 numerical_screen_rejections=int((eligible & ~accept).sum()),
                 rank_screen_rejections=int((~eligible).sum()),
                 minimum_singular_ratio_lower_bound=float(ratio_lower.min()),
                 maximum_heuristic_relative_roundoff=float(screen[eligible].max()) if eligible.any() else 0.,
                 maximum_accepted_heuristic_relative_roundoff=float(screen[accept].max()) if accept.any() else 0.,
                 maximum_all_node_heuristic_relative_roundoff=float(screen.max()),
                 quadrature_error_bounds=error, roundoff_certified=False)
    return value, stats
