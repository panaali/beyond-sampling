import numpy as np

# Notation used in labels: max(xa..xb) = largest of those inputs; "second-largest of x0..x8" likewise;
# a maximum "sits at an odd position" if its index (0-based) is odd.

LABELS = {
    0: ("A", "Running-maximum memory: about 63*max(x0..x8) minus a fading trace of recent inputs "
             "(15*x8, 11*x7, 8.5*x9, 4.6*each of x4,x5,x6). Always positive."),
    1: ("A", "How far x6 falls below a reference R: about 12*ReLU(R - x6), where R is roughly the "
             "second-largest of x0..x8 if the max of x0..x8 sits at an even position, else roughly that max."),
    2: ("A", "How far x9 falls below a reference R: about 13*ReLU(R - x9), where R is roughly max(x0..x8) "
             "if that max sits at an even position, else roughly the second-largest of x0..x8."),
    3: ("C", "Mixed: about 23*max(x0..x7), minus 6.5*that max if it sits at an odd position, minus "
             "6.3*min(x4, max(x0..x3)) and 3.8*x1, plus 4.2*x2. Always positive."),
    4: ("A", "Running maximum emphasising odd positions: about 20*max(x0..x7), plus 10.5*that max if it "
             "sits at an odd position, plus 6.5*total rise of the running max at x1,x3,x5,x7, minus 6.2*x6, 3.6*x4."),
    5: ("A", "Running-maximum memory: about 37*max(x0..x8), minus 7.7*x4 and 7.4*x8, plus "
             "7.6*ReLU(max(x0..x6) - x7) and 5.6*x9. Always positive."),
    6: ("A", "How far x7 falls below the maximum before it: about 13*ReLU(R - x7), R roughly max(x0..x7) "
             "if that max sits at an even position, else about 0.6*max + 0.4*second-largest of x0..x7."),
    7: ("A", "Running-maximum memory: about 23*max(x0..x8) + 21*max(x0..x7), minus 7.8*max(x0..x6) when "
             "that max sits at an odd position, minus 5.1*x7 and 3.2*x5. Always positive."),
    8: ("A", "Running maximum with emphasis on a new maximum at x8: about 21*max(x0..x8) + "
             "27*ReLU(x8 - max(x0..x7)) - 10.5*x8 + 7.8*max(x0..x7) if at odd position + 4.9*ReLU(max(x0..x4) - x5)."),
    9: ("A", "Memory of the early running maximum: about 17*max(x0..x7) minus 3.2*(x4+x5+x6), 2.8*x2 and "
             "3.9*min(x5, max(x0..x4)), plus 2.7*max(x0..x8) if that max sits at an odd position."),
    10: ("A", "Detects that the last input x9 is the overall maximum: about 16*ReLU(x9 - max(x0..x8)), "
              "the margin by which x9 beats every earlier input; zero otherwise."),
    11: ("A", "How far x9 falls below a reference R: about 11*ReLU(R - x9), where R is roughly the "
              "second-largest of x0..x8 if the max of x0..x8 sits at an even position, else roughly that max."),
    12: ("B", "Memory of x3 plus an early-rise pattern: ReLU of 11*x3 - 4.3*ReLU(x3 - max(x0..x2)) "
              "+ 5*ReLU(x0 - x1) + 5*ReLU(x2 - max(x0,x1)) - 1*ReLU(max(x0..x3) - x4)."),
    13: ("A", "Flags a new maximum at x8: about 20.5*max(x0..x7) + 51*ReLU(x8 - max(x0..x7)), minus small "
              "traces 2.3*(x4+x5+x6), 3.1*x2 and 3.1*ReLU(x1 - x0). Always positive."),
    14: ("A", "Detects the overall maximum sitting at x6: when x6 is the largest of x0..x9, about "
              "2.1*x6 + 1.4*(second-largest value) - 1.5*x9; otherwise zero."),
    15: ("C", "Mixed, partly understood: ReLU of 13*max(x0..x7) - 5.9*that max if at an odd position "
              "+ 6.7*x2 - 7.1*min(x5, max(x0..x4)) + 4.8*ReLU(x1 - x0)."),
}

SEL = {0: ['M8', 'x8', 'x7', 'x9', 'x456'], 1: ['M8', 'S8', 'M8o', 'S8o', 'x6'],
       2: ['M8', 'S8', 'M8o', 'S8o', 'x9'], 3: ['M7', 'M7o', 'min4', 'x2', 'x1'],
       4: ['M7', 'M7o', 'x6', 'Jodd7', 'x4'], 5: ['M8', 'x4', 'x8', 'dn7', 'x9'],
       6: ['M7', 'S7', 'M7o', 'S7o', 'x7'], 7: ['M8', 'M7', 'M6o', 'x5', 'x7'],
       8: ['M8', 'up8', 'M7o', 'x8', 'dn5'], 9: ['M7', 'x456', 'x2', 'min5', 'M8o'],
       10: ['up9'], 11: ['M8', 'S8', 'M8o', 'S8o', 'x9'],
       12: ['x3', 'dn1', 'up2', 'up3', 'dn4'], 13: ['up8', 'M7', 'x456', 'x2', 'up1'],
       14: ['g6x6', 'g6s1', 'g6x9'], 15: ['min5', 'x2', 'M7', 'M7o', 'up1']}

COEF = {0: [63.19, -15.34, -10.78, -8.52, -4.57], 1: [0.77, 10.71, 19.17, -17.52, -12.05],
        2: [17.72, -4.45, -16.32, 15.39, -12.95], 3: [22.73, -6.53, -6.34, 4.19, -3.83],
        4: [19.8, 10.53, -6.23, 6.5, -3.57], 5: [36.61, -7.72, -7.41, 7.58, 5.63],
        6: [14.75, -1.5, -6.93, 6.61, -12.88], 7: [22.6, 21.21, -7.83, -3.21, -5.1],
        8: [21.12, 26.62, 7.78, -10.48, 4.85], 9: [17.02, -3.19, -2.8, -3.93, 2.7],
        10: [15.98], 11: [3.21, 7.36, 9.72, -9.02, -10.96],
        12: [11.18, 5.02, 5.03, -4.29, -0.98], 13: [51.35, 20.49, -2.32, -3.14, -3.06],
        14: [2.07, 1.39, -1.49], 15: [-7.09, 6.72, 13.22, -5.92, 4.79]}

CONSTANTS = {i: len(COEF[i]) for i in range(16)}

_par = "parity split by whether argmax of the prefix is odd"
STRUCT = {
    0: "feature set chosen by greedy/swap search: max(x0..x8), x7, x8, x9, x4+x5+x6",
    1: "template ReLU(a*M+b*S+odd*(c*M+d*S)-e*x_t), searched prefix end k=8 and t=6; " + _par,
    2: "same template, k=8, t=9; " + _par,
    3: "greedy/swap search over max/parity/position features",
    4: "greedy/swap search; Jodd7 = sum of running-max increments at positions 1,3,5,7",
    5: "greedy/swap search",
    6: "same template as neuron 1, k=7 (prefix x0..x7), t=7; " + _par,
    7: "greedy/swap search",
    8: "greedy/swap search",
    9: "greedy/swap search",
    10: "single feature ReLU(x9 - max(x0..x8))",
    11: "same template as neuron 1, k=8, t=9; " + _par,
    12: "greedy/swap search over running-max rise/fall features",
    13: "greedy/swap search",
    14: "gate: argmax of x0..x9 == 6 (found from intermediate-state analysis)",
    15: "greedy/swap search",
}


def _pref(x, k):
    p = x[:, :k + 1]
    return p.max(1), np.sort(p, 1)[:, -2], p.argmax(1)


def _feats(x):
    x = np.asarray(x, dtype=float)
    F = {}
    M = np.maximum.accumulate(x, 1)
    for k in range(10):
        F['x%d' % k] = x[:, k]
    for k in (6, 7, 8):
        Mk, Sk, Ak = _pref(x, k)
        od = (Ak % 2 == 1).astype(float)
        F['M%d' % k] = Mk; F['S%d' % k] = Sk; F['M%do' % k] = Mk * od; F['S%do' % k] = Sk * od
    d = np.diff(np.c_[np.zeros(len(x)), M], axis=1)
    F['Jodd7'] = d[:, 1] + d[:, 3] + d[:, 5] + d[:, 7]
    for k in range(1, 10):
        F['dn%d' % k] = np.maximum(M[:, k - 1] - x[:, k], 0)
        F['up%d' % k] = np.maximum(x[:, k] - M[:, k - 1], 0)
        F['min%d' % k] = np.minimum(x[:, k], M[:, k - 1])
    F['x456'] = x[:, 4] + x[:, 5] + x[:, 6]
    g = (x.argmax(1) == 6).astype(float)
    s1 = np.sort(x, 1)[:, -2]
    F['g6x6'] = g * x[:, 6]; F['g6s1'] = g * s1; F['g6x9'] = g * x[:, 9]
    return F


def f(i, x):
    F = _feats(x)
    z = sum(c * F[n] for c, n in zip(COEF[i], SEL[i]))
    return np.maximum(z, 0.0)
