"""Counted T5 tetra integration; reuses T52 geometry without modifying it."""
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, '/workspace/clean-neurons')
from m16.approaches.small_ladder_review.portable_cost.accounting import Audit, gauss_rule
from m16.approaches.small_ladder_review.portable_cost.t5.geometry import enumerate_correct, determinant
from m16.approaches.small_ladder_review.portable_cost.t5.volume import bernstein, segment
from fixed_tangent import fixed_tangent_tetra
from primitive import radial_interval


def frame(edges, audit):
    count = len(edges)
    q = np.zeros_like(edges)
    good = np.ones(count, bool)
    for j in range(3):
        column = edges[:, :, j].copy()
        for repeat in range(2):
            for k in range(j):
                dot = np.sum(q[:, :, k]*column, axis=1)
                column -= dot[:, None]*q[:, :, k]
                audit.add('tetra_frame_projection', 27*count)  # 19 arithmetic + control
        norm = np.sqrt(np.sum(column*column, axis=1))
        valid = norm > 0
        q[:, :, j] = np.divide(column, norm[:, None], out=np.zeros_like(column), where=valid[:, None])
        good &= valid
        audit.add('tetra_frame_normalize', 48*count)  # dot9, sqrt20, divide5, tests/control
    gram = np.swapaxes(q, 1, 2) @ q
    good &= np.max(abs(gram-np.eye(3)), axis=(1, 2)) < 1e-10
    audit.add('tetra_frame_orthogonality', 128*count)  # 81 dot +27 tests +control
    audit.add('tetra_frame_control', 64*count)
    return q, good


def tail(a, b, degree, audit):
    """Positive nu=5/2 Chebyshev tail, finite products only."""
    audit.add('tetra_tail_setup', 100)  # two sqrt20, algebra, branches
    sa, sb = math.sqrt(a), math.sqrt(b)
    r = (sb-sa)/(sb+sa)
    if r == 0:
        return 0.
    c = (sa+sb)**2/4
    n = degree+1
    ratio = r*(2.5+n)/(n+1)
    if ratio >= 1:
        return float('inf')
    product = power = 1.
    for j in range(n):
        product *= (2.5+j)/(j+1)
        power *= r
        audit.add('tetra_tail_product', 8)
    z = 1-r*r*(2.5+n)/(n+1)
    audit.add('tetra_tail_finish', 80)  # two sqrt20 and <40 algebra/control
    return 2*product*power/(c*c*math.sqrt(c)*(1-ratio)*z*z*math.sqrt(z))


class PositiveFallback:
    def __init__(self, audit):
        self.audit = audit
        self.rules = {}

    def rule(self, n):
        if n not in self.rules:
            x, w = gauss_rule(n, self.audit)
            a, b = np.meshgrid(x, x, indexing='ij')
            wa, wb = np.meshgrid(w, w, indexing='ij')
            y = np.stack([a, (1-a)*b], axis=-1).reshape(-1, 2)
            rest = ((1-a)*(1-b)).ravel()
            weights = (wa*wb*(1-a)**2*(1-b)).ravel()
            self.audit.add('tetra_fallback_rule', 32*n*n)
            self.rules[n] = y, rest, weights
        return self.rules[n]

    def __call__(self, vertices, tolerance):
        lower, upper = radial_interval(vertices)
        self.audit.add('tetra_fallback_radial', 256)
        trials = 0
        for n in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128):
            trials += 1
            bound = tail(float(lower), float(upper), n-2, self.audit)/3
            self.audit.add('tetra_fallback_test', 16)
            if bound <= tolerance:
                break
        else:
            raise ArithmeticError('counted positive fallback tolerance unresolved')
        y, rest, weights = self.rule(n)
        prefix = y @ vertices[:2]
        a = prefix + rest[:, None]*vertices[2]
        b = prefix + rest[:, None]*vertices[3]
        self.audit.add('tetra_fallback_endpoints', 48*n*n)  # matrix15 + endpoints20 +control
        value = float(weights @ segment(a, b, self.audit))
        self.audit.add('tetra_fallback_sum', 2*n*n+32)
        return value, bound, n*n, trials


def primitive(rays, nodes, tolerance, audit, fallback):
    nc, nn = len(rays), len(nodes)
    # Conservative line-group bounds for fixed_tangent.py, excluding the
    # separately charged frame and positive fallback. See COUNTING.md.
    audit.add('tetra_cone_validation', 256*nc)
    audit.add('tetra_cone_coordinates', 256*nc)
    audit.add('tetra_cone_cross_products', 128*nc)
    audit.add('tetra_cone_inverse_control', 256*nc)
    audit.add('tetra_node_endpoints', 128*nc*nn)
    audit.add('tetra_node_norm_condition', 384*nc*nn)
    audit.add('tetra_node_facets', 448*nc*nn)
    audit.add('tetra_node_radial_screen', 384*nc*nn)
    audit.add('tetra_node_control', 128*nc*nn)
    return fixed_tangent_tetra(rays, nodes, absolute_tolerance=tolerance,
                               frame_builder=lambda edges: frame(edges, audit), fallback=fallback)


def volume(rays, audit, order=24, inner_budget=1e-9, batch=128):
    x, w = gauss_rule(order, audit)
    weights = w*(1-x)**3
    audit.add('tetra_outer_rule', 10*order+64)
    fallback = PositiveFallback(audit)
    total = error = outer = 0.
    counts = {}
    for start in range(0, len(rays), batch):
        v = np.asarray(rays[start:start+batch])
        determinants = np.array([abs(determinant(a, audit)) for a in v])
        factor = 3*determinants/(8*math.pi*math.pi)
        tolerance = inner_budget/(len(rays)*factor*weights.sum())
        value, stats = primitive(np.swapaxes(v, 1, 2), x, tolerance[:, None], audit, fallback)
        total += float(factor @ (value @ weights))
        error += float(factor @ (stats['quadrature_error_bounds'] @ weights))
        for ray, det in zip(v, determinants):
            outer += bernstein(ray, det, order, audit)
        audit.add('tetra_mass_sums', len(v)*(4*order+64)+32)
        for key in ('tetrahedra', 'analytic_tetrahedra', 'positive_fallbacks', 'fallback_segment_nodes', 'fallback_bound_trials', 'numerical_screen_rejections', 'rank_screen_rejections'):
            counts[key] = counts.get(key, 0)+stats[key]
    return total, dict(conditional_outer_bound=outer, positive_inner_bound=error, primitive_counts=counts)
