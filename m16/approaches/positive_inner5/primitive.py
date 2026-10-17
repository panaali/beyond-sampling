"""Frozen T11589 scalar primitive. Positive arithmetic is not an FP certificate."""
import math


class Ledger:
    def __init__(self):
        self.counts = {k: 0 for k in ('arithmetic', 'comparison_boolean', 'abs', 'sqrt', 'finite_check')}

    def op(self, name, *x):
        if name in ('add', 'sub', 'mul', 'div'):
            self.counts['arithmetic'] += 1
        elif name in ('eq', 'lt', 'gt', 'and', 'or', 'not'):
            self.counts['comparison_boolean'] += 1
        else:
            self.counts[name] += 1
        if name == 'add': return x[0] + x[1]
        if name == 'sub': return x[0] - x[1]
        if name == 'mul': return x[0] * x[1]
        if name == 'div': return x[0] / x[1]
        if name == 'eq': return x[0] == x[1]
        if name == 'lt': return x[0] < x[1]
        if name == 'gt': return x[0] > x[1]
        if name == 'and': return x[0] and x[1]
        if name == 'or': return x[0] or x[1]
        if name == 'not': return not x[0]
        if name == 'abs': return abs(x[0])
        if name == 'sqrt': return math.sqrt(x[0])
        if name == 'finite_check': return math.isfinite(x[0])
        raise ValueError(name)

    def total(self, x):
        out = x[0]
        for y in x[1:]: out = self.op('add', out, y)
        return out

    def all(self, x):
        out = x[0]
        for y in x[1:]: out = self.op('and', out, y)
        return out


def integrate(P, D):
    """Return integral_0^1 ||P+zD||^-5 dz or explicit unsupported status.

    Fixed five-element real float lists (not ndarray inputs). Nonconstant collinear segments remain
    unsupported even if their true integral is finite. No quadrature fallback.
    Common scaling and wedge delta reduce two cancellation mechanisms, but no
    directed rounding or total-return theorem is asserted.
    """
    L = Ledger(); o = L.op
    out = {'status': 'unresolved', 'value': None, 'branch': None, 'counts': L.counts}
    if type(P) is not list or type(D) is not list or len(P) != 5 or len(D) != 5:
        raise ValueError('five coordinates required; API shape checks outside numeric ledger')
    try:
        if o('not', L.all([o('finite_check', x) for x in P + D])):
            out['status'] = 'nonfinite_input'; return out
        constant = L.all([o('eq', x, 0.0) for x in D])
        magnitudes = [o('abs', x) for x in P + D]
        scale = magnitudes[0]
        for x in magnitudes[1:]:
            if o('gt', x, scale): scale = x
        if o('eq', scale, 0.0):
            out['status'] = 'origin'; return out
        p = [o('div', x, scale) for x in P]
        d = [o('div', x, scale) for x in D]
        a = L.total([o('mul', x, x) for x in d])
        c = L.total([o('mul', x, x) for x in p])
        out.update(scale=scale, a=a, c=c)
        if constant:
            out['branch'] = 'constant'
            if o('eq', c, 0.0):
                out['status'] = 'underflow_geometry'; return out
            j = o('div', 1.0, o('mul', o('mul', c, c), o('sqrt', c)))
        else:
            if o('eq', a, 0.0):
                out['status'] = 'underflow_geometry'; return out
            b = L.total([o('mul', x, y) for x, y in zip(p, d)])
            squares = []
            for i in range(5):
                for k in range(i + 1, 5):
                    w = o('sub', o('mul', p[i], d[k]), o('mul', p[k], d[i]))
                    squares.append(o('mul', w, w))
            delta = L.total(squares)
            out.update(b=b, delta=delta)
            if o('not', o('gt', delta, 0.0)):
                out['status'] = 'collinear_or_underflow'; return out
            t0 = b; t1 = o('add', a, b)
            r0sq = o('add', o('mul', t0, t0), delta)
            r1sq = o('add', o('mul', t1, t1), delta)
            r0 = o('sqrt', r0sq); r1 = o('sqrt', r1sq)
            ap = o('mul', a, o('sqrt', a))
            same = o('or', o('and', o('gt', t0, 0.0), o('gt', t1, 0.0)),
                     o('and', o('lt', t0, 0.0), o('lt', t1, 0.0)))
            if same:
                out['branch'] = 'same_sign'
                x0 = o('abs', t0); x1 = o('abs', t1)
                den = o('mul', o('mul', r0, r1), o('add', o('mul', x1, r0), o('mul', x0, r1)))
                V = o('div', o('mul', a, o('add', x0, x1)), den)
                inner = o('add', o('mul', 0.5, o('add', o('div', 1.0, r0sq), o('div', 1.0, r1sq))),
                          o('div', o('mul', delta, o('mul', V, V)), 6.0))
                j = o('mul', o('mul', ap, V), inner)
            else:
                out['branch'] = 'cross_or_endpoint'
                ds = o('add', o('abs', o('div', t0, r0)), o('abs', o('div', t1, r1)))
                inner = o('add', o('mul', 0.5, o('add', o('div', delta, r0sq), o('div', delta, r1sq))),
                          o('div', o('mul', ds, ds), 6.0))
                j = o('mul', o('div', ap, o('mul', delta, delta)), o('mul', ds, inner))
        scale2 = o('mul', scale, scale)
        scale5 = o('mul', o('mul', scale2, scale2), scale)
        value = o('div', j, scale5)
        out['value'] = value
        valid = o('and', o('finite_check', value), o('gt', value, 0.0))
        out['status'] = 'ok' if valid else 'nonfinite_or_underflow_result'
    except (OverflowError, ZeroDivisionError, ValueError) as exc:
        out.update(status='arithmetic_failure', exception=type(exc).__name__ + ': ' + str(exc))
    return out
