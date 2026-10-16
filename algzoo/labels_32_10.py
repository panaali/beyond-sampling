"""Neuron labels for the 2nd-argmax RNN (hidden 32, length 10).

Notation used in the labels (all computed from the input x of shape (N,10)):
  x9            the last input
  M8, S8, T8    largest, 2nd largest, 3rd largest of the first nine inputs x0..x8
  am8, a28      positions (0..8) of M8 and S8
  G = M8 - S8   gap between the top two of the first nine inputs ("max margin")
  G12 = S8 - T8 gap between 2nd and 3rd of the first nine
  top2_8(k)     indicator that x_k is among the top two of x0..x8
  T1_k, T2_k    largest / 2nd largest of x0..x8 excluding x_k
  relu(z) = max(z, 0)
  [cond]        indicator (0/1)
Neurons whose input weight is ~0 cannot see x9 at the last step, so their
labels are functions of x0..x8 only.

Recurring motif ("max-position code"): many neurons carry the max margin G
multiplied by a coefficient that depends on WHERE the max of x0..x8 occurred
(am8).  Each such neuron has its own set of positions for which the gain is
high; together these neurons form a distributed code for am8, scaled by G.
The sets below were found by search; they are part of the explanation.
"""
import numpy as np


def relu(z):
    return np.maximum(z, 0.0)


def _feats(x):
    x = np.asarray(x, dtype=float)
    p = x[:, :9]
    order = np.argsort(-p, axis=1)
    s = -np.sort(-p, axis=1)
    F = {}
    F['x'] = x
    F['x9'] = x[:, 9]
    F['M8'], F['S8'], F['T8'] = s[:, 0], s[:, 1], s[:, 2]
    F['am8'], F['a28'] = order[:, 0], order[:, 1]
    F['G'] = s[:, 0] - s[:, 1]
    F['G12'] = s[:, 1] - s[:, 2]
    rk = np.argsort(order, axis=1)
    F['rk8'] = rk  # rank of x_k among x0..x8 (0 = largest)
    # largest / 2nd largest of x0..x8 excluding x_k
    T1 = np.where(rk == 0, s[:, [1]], s[:, [0]])
    T2 = np.where(rk <= 1, s[:, [2]], s[:, [1]])
    F['T1'], F['T2'] = T1, T2
    return F


def _in(a, S):
    return np.isin(a, list(S)).astype(float)


def _formula(i, F, p):
    x9, M8, S8, G, G12 = F['x9'], F['M8'], F['S8'], F['G'], F['G12']
    am, a2, x, T1, T2, rk = F['am8'], F['a28'], F['x'], F['T1'], F['T2'], F['rk8']
    top2 = lambda k: (rk[:, k] <= 1).astype(float)
    if i in (0, 13, 29, 6, 18):
        S1, S2 = {0: ({1, 2, 5, 6}, {0, 2, 3, 4, 5, 6}), 13: ({1, 2, 4, 7}, {0, 2, 4, 6, 7}),
                  29: ({0, 3, 6, 7}, {3}), 6: ({0, 2, 5, 7}, {0, 2}), 18: ({1, 4, 6, 8}, {0, 1})}[i]
        out = relu(p[0] * (S8 - x9) + (G + p[1]) * (p[2] + p[3] * _in(am, S1) + p[4] * _in(am, S2)))
        if i in (0, 13, 29):
            out = out * (am != 8)
        return out
    if i == 7:
        return relu(p[0] * (x9 - M8) + p[1])
    if i == 16:
        return relu(p[0] * x9 - p[1] * S8 + p[2])
    if i == 12:
        return relu(p[0] * (S8 - x9) + p[1])
    if i == 21:
        return relu(p[0] * x9 - p[1] * S8 - G * (p[2] + p[3] * (am % 2 == 1)) + p[4])
    if i == 23:
        return relu(p[0] * x9 - p[1] * S8 - G * (p[2] + p[3] * _in(am, {2, 6, 7})) + p[4]) * (am != 8)
    if i == 1:
        return relu(p[0] * G + p[1]) * (am != 8)
    if i == 5:
        return relu((G + p[0]) * (p[1] + p[2] * _in(am, {0, 3, 4, 8}) + p[3] * _in(am, {1, 8})))
    if i == 17:
        return relu(p[0] * M8 + p[1] * S8 + p[2] * M8 * _in(am, {0, 1, 3, 4, 6, 7}) + p[3])
    if i == 31:
        return relu(p[0] * S8 + p[1] * G * _in(am, {0, 1, 3, 4}) + p[2] * M8 + p[3])
    if i == 27:
        return relu(_in(am, {2, 6}) * (p[0] + p[1] * G + p[2] * M8) + p[3])
    if i == 11:
        return relu(p[0] * M8 * _in(am, {0, 4, 6}) + p[1] * S8 * _in(am, {0, 3, 5, 6, 7, 8}) + p[2] * relu(x[:, 4] - T1[:, 4]) + p[3])
    if i == 10:
        return relu(_in(am, {4, 5}) * (p[0] + p[1] * M8 + p[2] * a2) + p[3])
    if i == 2:   # slot 3
        return relu(p[0] * top2(3) + p[1] * M8 * _in(am, {0, 3}) + p[2] * x[:, 3] + p[3] * F['T8'] + p[4])
    if i == 15:  # slot 1
        return relu(p[0] * relu(x[:, 1] - T2[:, 1]) + p[1] * top2(1) + p[2] * M8 * (am == 1) + p[3] * M8 * _in(a2, {0, 1}) + p[4])
    if i == 19:  # slot 6
        return relu(p[0] * top2(6) + p[1] * M8 * (am == 6) + p[2] * S8 * (a2 == 6) + p[3] * M8 * _in(am, {0, 2}) + p[4])
    if i == 22:  # slot 7
        return relu(p[0] * (am == 7) + p[1] * relu(x[:, 7] - T2[:, 7]) + p[2] * x[:, 7] + p[3] * S8 * (am != 7) + p[4])
    if i == 26:  # slot 2
        return relu(p[0] * relu(x[:, 2] - T2[:, 2]) + p[1] * M8 * _in(am, {0, 2, 4}) + p[2] * top2(2) + p[3] * G * _in(am, {0, 2, 7}) + p[4])
    if i == 14:  # slot 8
        return relu(p[0] * relu(T2[:, 8] - x[:, 8]) + p[1] * relu(x[:, 0] - T2[:, 0]) + p[2] * G * _in(am, {1, 2, 3, 4}) + p[3] * relu(T2[:, 4] - x[:, 4]) + p[4])
    if i == 24:  # slot 5 (negative)
        return relu(p[0] * relu(T2[:, 5] - x[:, 5]) + p[1] * top2(5) + p[2] * np.sort(x[:, :8], 1)[:, -2] + p[3] * relu(T1[:, 6] - x[:, 6]) + p[4])
    if i == 30:  # slot 4 (negative)
        return relu(p[0] * relu(T1[:, 4] - x[:, 4]) + p[1] * S8 + p[2] * top2(4) + p[3] * M8 * _in(am, {0, 3}) + p[4])
    if i == 20:  # slot 0 (negative)
        return relu(p[0] * relu(T1[:, 0] - x[:, 0]) + p[1] * top2(0) + p[2] * M8 * _in(am, {1, 8}) + p[3] * G12 * _in(a2, {1, 2, 4, 6, 8}) + p[4])
    if i == 25:  # slot 8 (negative)
        return relu(p[0] * top2(8) + p[1] * S8 * _in(a2, {0, 1, 2, 3, 4, 5, 6}) + p[2] * relu(T1[:, 8] - x[:, 8]) + p[3] * S8 * _in(a2, {0, 1, 2, 3, 4, 7, 8}) + p[4])
    if i == 28:
        return relu(M8 * (p[0] * _in(am, {0, 1, 2, 3, 4, 5, 7}) + p[1] * _in(am, {0, 1, 3, 6})) + p[2] * relu(T2[:, 8] - x[:, 8]) + p[3] * _in(am, {0, 1, 2, 4, 5}) + p[4])
    if i == 3:
        return relu(p[0] * G * _in(am, {0, 1, 2, 3, 4}) + p[1] * _in(a2, {0, 1, 2, 3, 4, 5, 8}) + p[2] * top2(8) + p[3] * M8 * _in(am, {0, 3, 8}) + p[4])
    if i == 4:
        return relu(p[0] * S8 * _in(a2, {0, 2, 4, 6}) + p[1] * _in(a2, {1, 5, 7}) + p[2] * M8 * _in(am, {1, 2, 4}) + p[3] * G12 * _in(a2, {0, 4, 5, 6, 8}) + p[4])
    if i == 8:
        return relu(p[0] * M8 * _in(am, {0, 1, 2, 5}) + p[1] * relu(x[:, 5] - T2[:, 5]) + p[2] * np.max(x[:, :5], 1) + p[3] * F['T8'] + p[4])
    if i == 9:
        return relu(p[0] * S8 + p[1] * np.sort(x[:, :9], 1)[:, -4] + p[2] + p[3] * relu(x[:, 8] - T2[:, 8]))
    raise ValueError(i)


P = {
    0: [1.5, 0.311, -0.205, 0.923, 0.351],
    1: [4.419, 0.3684],
    2: [1.556, 0.7093, 1.812, -1.166, 0.3251],
    3: [1.157, -1.025, -3.795, 0.4411, 1.162],
    4: [0.5238, -0.3176, 0.1533, 0.3174, 0.3519],
    5: [0.387, 2.576, 1.341, 0.979],
    6: [1.798, 0.16, 0.151, 0.926, 0.802],
    7: [8.769, 0.06148],
    8: [0.77, 1.84, 0.8518, -0.5789, 2.008],
    9: [-2.706, -0.8402, -0.1389, 0.2289],
    10: [0.5382, 0.3546, -0.1286, 0.01593],
    11: [0.9165, -0.5176, 0.7087, 0.2272],
    12: [3.082, 0.06899],
    13: [1.659, 0.355, 0.026, 1.024, 0.245],
    14: [-6.385, 1.857, 1.037, 0.3653, 1.421],
    15: [1.454, 1.232, 1.701, 0.409, 0.1972],
    16: [5.689, 5.427, -0.1364],
    17: [8.371, 2.054, -0.4224, 0.02491],
    18: [1.345, 0.303, -0.278, 0.771, 1.358],
    19: [1.157, 1.667, 1.072, 0.2234, 0.1441],
    20: [0.3962, -1.329, 0.5133, 0.918, 0.7798],
    21: [4.347, 4.23, 1.502, 2.294, -0.1001],
    22: [0.6281, 0.6328, 1.411, -1.426, 0.6481],
    23: [4.847, 4.617, 0.6081, 0.9443, -0.1577],
    24: [0.3902, -2.928, 0.684, 0.186, 0.4292],
    25: [-2.822, 0.6155, 0.1934, 0.3043, 0.1918],
    26: [1.625, 1.21, 1.588, 1.162, 0.5847],
    27: [0.4719, 0.7352, 0.6411, 0.02078],
    28: [0.5501, 0.3974, 0.2098, 0.2719, 0.1536],
    29: [1.621, 0.445, 0.004, 0.832, 0.769],
    30: [0.5316, 0.8916, -2.058, 0.3769, 0.4343],
    31: [5.844, 1.532, 0.1248, 0.4754],
}

LABELS = {}


def f(i, x):
    return _formula(i, _feats(x), P[i])
