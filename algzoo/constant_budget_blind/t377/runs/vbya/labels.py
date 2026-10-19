import numpy as np

# ---------------------------------------------------------------- helpers
def _R(z):
    return np.maximum(z, 0.0)

def _top(a, k):
    """k-th largest (1-based) along axis 1."""
    return -np.sort(-a, axis=1)[:, k - 1]

def _pre(x):
    p = x[:, :9]
    o = np.argsort(-p, axis=1)
    return _top(p, 1), _top(p, 2), o[:, 0], o[:, 1]   # ps1, ps2, pam, pa2 over x0..x8

def _po(x, t, k):
    """k-th largest of x0..x8 excluding x_t."""
    return _top(np.delete(x[:, :9], t, axis=1), k)

def _o(x, t, k):
    """k-th largest of x0..x9 excluding x_t."""
    return _top(np.delete(x, t, axis=1), k)

def _mx(x, lo, hi, excl=None):
    """max of x_lo..x_hi (inclusive), optionally excluding position excl."""
    idx = [j for j in range(lo, hi + 1) if j != excl]
    return x[:, idx].max(axis=1)

def _D(x, t):
    """lead of x_t over every other value of x0..x8 (0 unless x_t is the max of x0..x8)."""
    return _R(x[:, t] - _po(x, t, 1))

def _G(x, T):
    """gap between largest and second-largest of x0..x8, counted only if the largest sits in T."""
    ps1, ps2, pam, _ = _pre(x)
    return np.isin(pam, T) * (ps1 - ps2)

# ---------------------------------------------------------------- family formulas
# x9 family: relu(w*x9 + a*ps1 + b*ps2 + c*G_T + e)
X9 = {
    3:  (-3.972, [1, 4, 7],              [-0.138, 4.140, 5.286, 0.034]),
    7:  (-5.053, [8],                    [5.024, 0.041, -5.193, 0.067]),
    9:  (-4.479, [0, 1, 4, 5, 6, 7, 8],  [0.843, 3.563, 3.339, -0.123]),
    17: (-6.038, [0, 1, 2, 3],           [0.020, 6.089, 4.304, 0.117]),
    18: (-4.416, [0, 5, 7, 8],           [-0.190, 4.590, 4.855, 0.049]),
    20: (-4.684, [0, 3, 4],              [-0.007, 4.670, 5.849, -0.018]),
    22: (-4.067, [0, 1, 2, 4, 5],        [4.604, -0.515, -4.330, 0.033]),
    28: (-3.657, [0, 1, 4, 5, 6, 8],     [-0.298, 3.938, 4.877, 0.042]),
    30: (-4.354, [1, 3, 5],              [4.531, -0.154, -4.901, 0.082]),
    8:  (6.877,  [0, 1, 2],              [-6.752, -0.059, 1.270, 0.039]),
    13: (5.972,  [0, 1],                 [-0.100, -5.816, -4.765, -0.029]),
    19: (0.952,  [0, 1],                 [10.086, 4.838, 3.759, 0.795]),
}

# step-t family: relu(w*x_t + a*q1 + b*q2 + c*[argmax of x0..x_{t-1} in T]*(q1-q2) + e),
# q1, q2 = largest / second-largest of x0..x_{t-1}
TFAM = {
    14: (8, [1, 2, 3], [-5.503, 2.830, 2.563, 2.513, 0.223]),
    27: (8, [0],       [8.299, -7.980, -0.089, 1.774, 0.182]),
}

def _tfam(x, t, T, c):
    p = x[:, :t]
    q1, q2 = _top(p, 1), _top(p, 2)
    am = np.argmax(p, axis=1)
    D = np.isin(am, T) * (q1 - q2)
    return _R(c[0] * x[:, t] + c[1] * q1 + c[2] * q2 + c[3] * D + c[4])

def _Q2(x, t):
    return _top(x[:, :t], 2)

def _lin(cols, c, relu=True):
    z = sum(ci * col for ci, col in zip(c[:-1], cols)) + c[-1]
    return _R(z) if relu else z

def f(i, x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if i in X9:
        w, T, c = X9[i]
        ps1, ps2, _, _ = _pre(x)
        return _R(w * x[:, 9] + c[0] * ps1 + c[1] * ps2 + c[2] * _G(x, T) + c[3])
    if i in TFAM:
        t, T, c = TFAM[i]
        return _tfam(x, t, T, c)
    ps1, ps2, pam, pa2 = _pre(x)
    if i in (21, 1, 11):
        t = {21: 1, 1: 2, 11: 3}[i]
        c = {21: [13.615, 4.610, 1.428, 2.283, 0.319],
             1: [8.976, 3.155, 1.112, 3.332, 0.655],
             11: [7.894, 2.854, 1.011, 4.225, 0.312]}[i]
        cols = [_D(x, t), _R(x[:, t] - _o(x, t, 3)), (pa2 == t) * 1.0, _D(x, 0)]
        return _lin(cols, c)
    if i == 2:
        cols = [_D(x, 4), _R(x[:, 4] - _o(x, 4, 3)), _D(x, 0), _R(x[:, 4] - _Q2(x, 4))]
        return _lin(cols, [5.956, 2.562, 4.373, 1.120, 0.202])
    if i == 23:
        cols = [_D(x, 6), (pam == 6) * 1.0, (pa2 == 6) * 1.0, _R(x[:, 6] - _Q2(x, 6))]
        return _lin(cols, [2.915, 3.686, 2.334, 2.902, 0.120])
    if i == 6:
        cols = [_G(x, [0, 1, 2, 3]), _G(x, [4, 5, 6]), ps1, ps2]
        return _lin(cols, [-8.002, -2.790, 5.708, -5.566, 0.064])
    if i == 4:
        cols = [_G(x, [0, 1]), _G(x, [2, 8]), ps1, ps2]
        return _lin(cols, [-5.793, -2.963, 6.654, -6.044, 0.276])
    if i == 15:
        cols = [_R(x[:, 0] - _mx(x, 1, 6)), _D(x, 1), _D(x, 2), _D(x, 3)]
        return _lin(cols, [16.03, 13.01, 9.94, 7.52, 3.29], relu=False)
    if i == 0:
        cols = [_R(x[:, 0] - _mx(x, 1, 4)), _R(x[:, 1] - _mx(x, 0, 4, excl=1)),
                _R(_po(x, 1, 1) - x[:, 1]), _R(x[:, 0] - _mx(x, 1, 6))]
        return _lin(cols, [6.08, 5.70, 1.28, 6.85, 2.93], relu=False)
    if i == 29:
        cols = [np.maximum(x[:, 0], x[:, 1]), (pa2 == 0) * 1.0, _R(x[:, 0] - _mx(x, 1, 5)), (pa2 == 1) * 1.0]
        return _lin(cols, [1.06, 3.07, 2.58, 1.43, -0.22])
    if i == 5:
        cols = [_mx(x, 1, 6), _D(x, 2), x[:, 0], _D(x, 7)]
        return _lin(cols, [0.65, 1.72, -0.35, -99.89, 0.03])
    if i == 12:
        cols = [_G(x, [2, 3, 4, 6, 7]), _R(_po(x, 5, 1) - x[:, 5]), _R(x[:, 5] - _po(x, 5, 2)), ps2]
        return _lin(cols, [5.20, 0.85, -3.13, 1.16, 0.76])
    if i == 26:
        cols = [_R(_Q2(x, 7) - x[:, 7]), _R(x[:, 0] - _mx(x, 1, 5)), _R(x[:, 7] - _Q2(x, 7)), _R(x[:, 8] - _po(x, 8, 2))]
        return _lin(cols, [-8.39, 7.15, 3.37, -1.56, 2.44])
    if i == 31:
        cols = [_R(x[:, 8] - _po(x, 8, 2)), _mx(x, 1, 8), _R(_Q2(x, 8) - x[:, 8]), _R(_mx(x, 0, 7) - x[:, 8])]
        return _lin(cols, [-8.53, 0.86, 0.76, -0.55, 0.26])
    if i == 10:
        cols = [_R(x[:, 0] - _po(x, 0, 2)), _R(x[:, 1] - _po(x, 1, 2)), _mx(x, 0, 5)]
        return (pam == 8) * _lin(cols, [4.878, 3.995, 0.818, 0.018])
    if i == 24:
        cols = [_R(_mx(x, 0, 6) - x[:, 7]), _D(x, 0), _R(x[:, 1] - _mx(x, 0, 7, excl=1)), _R(x[:, 8] - _po(x, 8, 2))]
        return _lin(cols, [2.43, 4.28, 2.16, 1.23, 0.71], relu=False)
    if i in (16, 25):
        return np.zeros(N)
    raise ValueError(i)



_x9ref = lambda S, k: ("x9 falls short of a reference r: about %s*max(0, r - x9), where r is the largest of x0..x8 "
                       "if that maximum sits at %s, otherwise the second-largest of x0..x8." % (k, S))
LABELS = {
    0: ("B", "Early-record detector: grows with x0's lead over x1..x4 and over x1..x6, and with x1's lead over x0,x2..x4; "
             "slightly larger when x1 is well below the maximum of the other values; baseline about 3."),
    1: ("A", "x2 is the largest of x0..x8: about 9*(x2 minus the largest other of x0..x8, if positive) + 3*(x2's lead over "
             "the third-largest other value) + bonus if x2 is runner-up; plus 3*x0's lead when x0 is the max."),
    2: ("A", "x4 is the largest of x0..x8: about 6*(x4 minus the largest other of x0..x8, if positive) + 2.6*(x4's lead over "
             "the third-largest other value) + 1.1*(x4 above second-largest of x0..x3); plus 4.4*x0's lead when x0 is max."),
    3: ("A", _x9ref("x1, x4 or x7", 4)),
    4: ("A", "Gap between largest and second-largest of x0..x8, scaled by where the max sits: about 6.5*gap if at x3..x7, "
             "about 3.7*gap if at x2 or x8, near 0 if at x0 or x1."),
    5: ("C", "Weak neuron: about 0.65*max(x1..x6) - 0.35*x0 + 1.7*(x2's lead when x2 is the max of x0..x8), all under a ReLU, "
             "and silenced when x7 is the maximum of x0..x8."),
    6: ("A", "Gap between largest and second-largest of x0..x8 when the max is late: about 5.6*gap if the max is x7 or x8, "
             "about 2.9*gap if x4..x6, about 0 if x0..x3."),
    7: ("A", "About 5*max(0, max(x0..x7) - x9): how far x9 falls below the largest of x0..x7."),
    8: ("A", "About 6.9*max(0, x9 - max(x0..x8)): x9 is the new overall maximum, and by how much (slightly more when the "
             "earlier maximum was at x0..x2)."),
    9: ("A", _x9ref("any position except x2 or x3", 4.4)),
    10: ("B", "Active only when x8 is the largest of x0..x8; then about 4.9*x0's and 4*x1's margin over the third-largest "
              "(x0 or x1 being runner-up) + 0.8*max(x0..x5)."),
    11: ("A", "x3 is the largest of x0..x8: about 8*(x3 minus the largest other of x0..x8, if positive) + 2.9*(x3's lead over "
              "the third-largest other value) + bonus if x3 is runner-up; plus 4*x0's lead when x0 is the max."),
    12: ("B", "About 5.2*gap between largest and second-largest of x0..x8 when the max sits at x2,x3,x4,x6 or x7, minus about "
              "3*x5's margin over the third-largest when x5 is in the top two; plus 1.2*second-largest."),
    13: ("A", "About 6*max(0, x9 - r): x9 beats r, the second-largest of x0..x8 raised most of the way to the largest when "
              "that maximum is at x0 or x1. Signals x9 is in the top two."),
    14: ("A", "About 5.5*max(0, r - x8), r roughly the average of the largest and second-largest of x0..x7 (moved to the "
              "largest when it sits at x1..x3): x8 is not a new running maximum."),
    15: ("A", "Lead of an early maximum: about 16*(x0 - max(x1..x6), if positive) + 13, 10, 7.5 times the lead of x1, x2, x3 "
              "over all other x0..x8; baseline about 3.3."),
    16: ("A", "Dead: essentially always 0 on Gaussian inputs."),
    17: ("A", _x9ref("x0..x3", 6)),
    18: ("A", _x9ref("x0, x5, x7 or x8", 4.4)),
    19: ("A", "Size of the top values, always active: about 10*(largest of x0..x8) + 5*(second-largest of x0..x8) + x9, "
              "plus 4*their gap when the max is at x0 or x1."),
    20: ("A", _x9ref("x0, x3 or x4", 4.7)),
    21: ("A", "x1 is the largest of x0..x8: about 14*(x1 minus the largest other of x0..x8, if positive) + 4.6*(x1's lead over "
              "the third-largest other value) + bonus if x1 is runner-up; plus 2.3*x0's lead when x0 is max."),
    22: ("A", _x9ref("x3, x6, x7 or x8", 4.1)),
    23: ("A", "x6 is among the top two of x0..x8: bonuses if x6 is the max or runner-up, plus about 3*x6's lead over the "
              "other values and 3*x6's excess over the second-largest of x0..x5."),
    24: ("B", "x7 below the running maximum of x0..x6 (about 2.4*shortfall), plus 4.3*x0's lead when x0 is the max, 2.2*x1's "
              "lead over x0..x7, 1.2*x8's margin over the third-largest; baseline 0.7."),
    25: ("A", "Dead: essentially always 0 on Gaussian inputs."),
    26: ("B", "x7 versus the second-largest of x0..x6: rises 3.4*(amount above), falls 8.4*(amount below); plus 7*x0's lead "
              "over x1..x5; small drop if x8 is in the top two; ReLU, baseline 2.4."),
    27: ("A", "About 8.3*max(0, x8 - max(x0..x7)): x8 sets a new running maximum, and by how much."),
    28: ("A", _x9ref("x0, x1, x4, x5, x6 or x8", 3.8)),
    29: ("C", "x0 near the top: about max(x0,x1) + 3 if x0 is runner-up of x0..x8 + 2.6*x0's lead over x1..x5 + 1.4 if x1 is "
              "runner-up; ReLU."),
    30: ("A", _x9ref("any position except x1, x3 or x5", 4.4)),
    31: ("D", "Weak neuron, roughly 0.86*max(x1..x8) + 0.76*(second-largest of x0..x7 minus x8, if positive), shut off when "
              "x8 is in the top two of x0..x8; ReLU."),
}
CONSTANTS = {i: 5 for i in range(32)}
CONSTANTS[10] = 4
CONSTANTS[16] = 0
CONSTANTS[25] = 0
_s9 = "position set T (where the max of x0..x8 sits) searched greedily; features ps1, ps2, gap*[argmax in T], x9 fixed family; x9 slope measured by probing"
_sg = "features chosen by greedy forward search from a library of ~180 rank/record features (relu or linear chosen by test R2)"
STRUCT = {i: _sg for i in range(32)}
for _i in X9:
    STRUCT[_i] = _s9
for _i in TFAM:
    STRUCT[_i] = "step-t family (t=8): position set T of the max of x0..x7 searched greedily"
STRUCT[4] = STRUCT[6] = "two position sets T1, T2 for gap*[argmax of x0..x8 in T] searched by coordinate descent"
STRUCT[12] = "position set for gap feature searched; other features from greedy search"
STRUCT[10] = "gate [argmax of x0..x8 == 8] chosen from position gates; features from greedy search"
STRUCT[16] = STRUCT[25] = ""
