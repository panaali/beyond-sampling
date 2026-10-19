import numpy as np

def _relu(v):
    return np.maximum(v, 0.0)

def _feat(name, x):
    """Interpretable input features. W = x0..x8 (first nine values)."""
    x = np.asarray(x, dtype=float)
    W = x[:, :9]
    s = -np.sort(-W, 1)
    M9, S9, T9 = s[:, 0], s[:, 1], s[:, 2]          # largest, 2nd, 3rd of x0..x8
    o9 = np.argsort(-W, 1); am9, a29 = o9[:, 0], o9[:, 1]   # positions of max / 2nd of x0..x8
    oA = np.argsort(-x, 1); amA = oA[:, 0]          # position of max of all ten
    x9 = x[:, 9]
    g12 = M9 - S9
    n8 = 1.0 - (am9 == 8)
    simple = {
        'M9': lambda: M9, 'S9': lambda: S9, 'T9': lambda: T9, 'x9': lambda: x9, 'g12': lambda: g12,
        'r(S9-x9)': lambda: _relu(S9 - x9), 'r(M9-x9)': lambda: _relu(M9 - x9),
        'r(x9-M9)': lambda: _relu(x9 - M9), 'r(x9-S9)': lambda: _relu(x9 - S9), 'x9-M9': lambda: x9 - M9,
        'r(S9-x9)n8': lambda: _relu(S9 - x9) * n8, 'g12n8': lambda: g12 * n8,
        'A_xso9': lambda: x9 - S9, 'A_earl9': lambda: x9 - M9, 'A_rxso9': lambda: _relu(x9 - S9),
        'A_min': lambda: x.min(1), 'A_am': lambda: amA * 1.0, 'pos_a29': lambda: a29 * 1.0,
        'a29_even': lambda: (a29 % 2 == 0) * 1.0, 'am9_even': lambda: (am9 % 2 == 0) * 1.0,
        'am[2,6]': lambda: ((am9 == 2) | (am9 == 6)) * 1.0,
    }
    if name in simple:
        return simple[name]()
    if name == 'rmo[2,6]':
        return _feat('r(x2-mo)', x) + _feat('r(x6-mo)', x)
    if name.startswith('am') and name.endswith('*g12'):
        return (am9 == int(name[2])) * g12
    if name.startswith('amA'):
        return (amA == int(name[3])) * 1.0
    if name.startswith('A_P2_'):                  # second largest of x0..x_{k-1}
        return -np.sort(-x[:, :int(name[5:])], 1)[:, 1]
    if name.startswith('A_P'):                    # max of x0..x_{k-1}
        return x[:, :int(name[3:])].max(1)
    if name.startswith('A_earl'):                 # x_j minus max of earlier values
        j = int(name[6:]); return x[:, j] - x[:, :j].max(1)
    if name.startswith('A_rearl'):
        j = int(name[7:]); return _relu(x[:, j] - x[:, :j].max(1))
    if name.startswith('A_rxmo') or name.startswith('A_rxso'):   # vs max / 2nd of the other nine of all ten
        j = int(name[6:]); so = -np.sort(-np.delete(x, j, 1), 1)
        return _relu(x[:, j] - so[:, 0 if name[4:6] == 'mo' else 1])
    for pre in ('top2_', 'am', 'a2'):              # rank of x_j within x0..x8
        if name.startswith(pre) and name[len(pre):].isdigit():
            j = int(name[len(pre):]); rank = (W > W[:, [j]]).sum(1)
            return {'am': (rank == 0), 'a2': (rank == 1), 'top2_': (rank <= 1)}[pre] * 1.0
    # comparisons of x_j (j<9) with max (mo), 2nd (so), 3rd (to) of the OTHER values in x0..x8, or later values (lat)
    rel = name.startswith('r(')
    body = name[2:-1] if rel else name
    a, b = body.split('-')
    if a.startswith('x'):
        j = int(a[1:]); key = b; sign = 1
    else:
        j = int(b[1:]); key = a; sign = -1
    if key == 'lat':
        ref = W[:, j + 1:].max(1)
    else:
        so = -np.sort(-np.delete(W, j, 1), 1); ref = so[:, {'mo': 0, 'so': 1, 'to': 2}[key]]
    v = sign * (W[:, j] - ref)
    return _relu(v) if rel else v
