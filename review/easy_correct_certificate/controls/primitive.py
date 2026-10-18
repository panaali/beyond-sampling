"""Exact rational ownership/bookkeeping only. No network or probability evaluator."""
from fractions import Fraction as F


def rational(x):
    if type(x) not in (int, str, F):
        raise ValueError('exact integer/string/Fraction required')
    return F(x)


def ownership(box, gap):
    b = [[rational(x) for x in pair] for pair in box]
    g = rational(gap)
    if len(b) < 3 or g < 0 or any(len(p) != 2 or p[0] >= p[1] for p in b):
        raise ValueError('positive-width T>=3 box and nonnegative gap required')
    inner, outer = [], []
    for i in range(len(b)):
        for j in range(len(b)):
            if i == j:
                continue
            others = [r for r in range(len(b)) if r not in (i, j)]
            inside = b[i][0] - b[j][1] >= g and b[j][0] - max(b[r][1] for r in others) >= g
            if inside:
                inner.append([i, j])
            lo = max(b[j][0], max(b[r][0] for r in others) + g)
            hi = min(b[j][1], b[i][1] - g)
            if lo <= hi:
                x = [p[0] for p in b]
                x[j] = lo
                x[i] = max(b[i][0], lo + g)
                outer.append(dict(pair=[i, j], second_interval=[lo, hi], witness=x))
    return dict(box=b, gap=g, inner_pairs=inner, outer_pairs=outer, inner=bool(inner), outer=bool(outer))


def margin_status(target, intervals):
    if type(target) is not int or len(intervals) < 2 or not 0 <= target < len(intervals):
        raise ValueError('invalid target')
    v = [[rational(x) for x in p] for p in intervals]
    if any(len(p) != 2 or p[0] > p[1] for p in v) or v[target] != [0, 0]:
        raise ValueError('invalid margin intervals or self contrast')
    correct = all(o == target or (p[0] > 0 if o < target else p[0] >= 0) for o, p in enumerate(v))
    wrong = any(o != target and (p[1] <= 0 if o < target else p[1] < 0) for o, p in enumerate(v))
    if correct and wrong:
        raise AssertionError('contradictory whole-domain certificate')
    return dict(target=target, margins=v, status='correct' if correct else 'wrong' if wrong else 'unknown')


def mass_bounds(event, correct, wrong):
    p, c, w = [[rational(x) for x in pair] for pair in (event, correct, wrong)]
    if any(len(v) != 2 or not 0 <= v[0] <= v[1] <= 1 for v in (p, c, w)):
        raise ValueError('invalid probability interval')
    if c[0] + w[0] > p[1] or c[0] > p[1] or w[0] > p[1]:
        raise ValueError('inconsistent disjoint-event bounds')
    unresolved = [max(F(0), p[0] - c[1] - w[1]), min(F(1), p[1] - c[0] - w[0])]
    if unresolved[0] > unresolved[1]:
        raise ValueError('inconsistent unresolved interval')
    conditional = None if p[0] == 0 else [c[0] / p[1], 1 - w[0] / p[1]]
    return dict(event=p, correct=c, wrong=w, unresolved=unresolved, conditional_accuracy=conditional)
