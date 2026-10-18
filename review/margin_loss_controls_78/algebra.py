"""Exact finite controls; no numerical work at import."""
from fractions import Fraction as Q


def matrix(a):
    if not a or not a[0] or any(len(r) != len(a[0]) for r in a):
        raise ValueError('nonempty rectangular matrix required')
    return [[Q(x) for x in r] for r in a]


def transpose(a):
    return [list(r) for r in zip(*a)]


def mul(a, b):
    if len(a[0]) != len(b):
        raise ValueError('matrix dimensions')
    return [[sum((x*y for x, y in zip(r, c)), Q(0))
             for c in transpose(b)] for r in a]


def centered(a):
    return [[x-sum(r, Q(0))/len(r) for x in r] for r in a]


def labels(a):
    return [max(range(len(r)), key=lambda j: r[j]) for r in a]


def solve(a, b, recorder):
    """Recorded Gauss-Jordan elimination with deterministic first pivot."""
    a, b = matrix(a), matrix(b)
    n = len(a)
    if len(a[0]) != n or len(b) != n:
        raise ValueError('square system required')
    entry = {'matrix': a, 'rhs': b, 'status': 'started'}
    recorder(entry)
    aug = [a[i][:]+b[i][:] for i in range(n)]
    try:
        for j in range(n):
            pivot = next((i for i in range(j, n) if aug[i][j]), None)
            if pivot is None:
                raise ValueError('singular system')
            aug[j], aug[pivot] = aug[pivot], aug[j]
            p = aug[j][j]
            aug[j] = [v/p for v in aug[j]]
            for i in range(n):
                if i != j:
                    q = aug[i][j]
                    aug[i] = [u-q*v for u, v in zip(aug[i], aug[j])]
        out = [r[n:] for r in aug]
        entry.update(status='returned', coefficients=out)
        recorder(None)
        return out
    except BaseException as exc:
        entry.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        recorder(None)
        raise


def ridge(x, y, penalty, recorder):
    x, y = matrix(x), matrix(y)
    n, d = len(x), len(x[0])
    if len(y) != n or len(penalty) != d or any(Q(v) < 0 for v in penalty):
        raise ValueError('ridge domain')
    a = mul(transpose(x), x)
    for j, p in enumerate(penalty):
        a[j][j] += Q(p)
    return solve(a, mul(transpose(x), y), recorder)


def pair_fit(x, y, penalty, convention, sample_mean, matched, recorder):
    """Build the pair design directly in the zero-sum coefficient gauge."""
    x, y = matrix(x), matrix(y)
    n, d, k = len(x), len(x[0]), len(y[0])
    if k < 2 or len(y) != n or len(penalty) != d or any(Q(v) < 0 for v in penalty):
        raise ValueError('pair domain')
    if convention not in ('unordered_sum', 'ordered_sum', 'unordered_mean', 'ordered_mean'):
        raise ValueError('pair convention')
    ordered = convention.startswith('ordered_')
    pairs = [(a, b) for a in range(k) for b in range(k)
             if (a != b if ordered else a < b)]
    weight = Q(1, len(pairs)) if convention.endswith('_mean') else Q(1)
    alpha = Q(2, k-1) if convention.endswith('_mean') else Q(2*k if ordered else k)
    sample_factor = Q(1, n) if sample_mean else Q(1)
    weight *= sample_factor
    v = [[Q(int(j == c))-Q(int(c == k-1)) for c in range(k)] for j in range(k-1)]
    dim = d*(k-1)
    a = [[Q(0) for _ in range(dim)] for _ in range(dim)]
    b = [[Q(0)] for _ in range(dim)]
    for xi, yi in zip(x, y):
        for c, e in pairs:
            feature = [xi[r]*(v[j][c]-v[j][e]) for r in range(d) for j in range(k-1)]
            target = yi[c]-yi[e]
            for u in range(dim):
                b[u][0] += weight*feature[u]*target
                for w in range(dim):
                    a[u][w] += weight*feature[u]*feature[w]
    penalty_factor = alpha*sample_factor if matched else Q(1)
    for r in range(d):
        for j in range(k-1):
            for l in range(k-1):
                a[r*(k-1)+j][r*(k-1)+l] += penalty_factor*Q(penalty[r])*sum(
                    (v[j][c]*v[l][c] for c in range(k)), Q(0))
    flat = solve(a, b, recorder)
    theta = [[flat[r*(k-1)+j][0] for j in range(k-1)] for r in range(d)]
    return {'coefficients': mul(theta, v), 'pairs': pairs, 'data_weight': weight,
            'alpha': alpha, 'sample_factor': sample_factor, 'penalty_factor': penalty_factor,
            'gauge': v, 'theta': theta}
