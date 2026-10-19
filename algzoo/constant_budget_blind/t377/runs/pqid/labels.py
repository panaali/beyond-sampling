import numpy as np

# Notation used in formulas (positions 0-based):
#  "prefix" = x0..x8 (everything except the last input x9).
#  M, S, T = largest, second-largest, third-largest of the prefix; am, a2 = their positions.
#  For "R"-type neurons, v1,v2,v3 / p1,p2,p3 = top three values / positions over all of x0..x9.

LABELS = {
    0: ("A", "Margin detector: (largest minus second-largest of x0..x8) weighted by where the largest sits (strong for x2-x4, silent for x7), plus a smaller term for (second minus third largest) and the third largest."),
    1: ("A", "Fires when x9 is below a running 'top value' of x0..x8: 10.3*ReLU(S + w*(M-S) - x9), M,S = two largest of x0..x8, w above 1 when M sits at x2-x4, about 0.7 at x8, about 0.2 at x0, x6, x7."),
    2: ("A", "Running top-value memory of x0..x8, ignoring x9: about 18*(S + w*(M-S)), M,S = two largest of x0..x8; weight w on the largest is highest when it sits at x5-x8, lower at x1-x2."),
    3: ("C", "Position-pattern code of the large values: grows with the top three values of x0..x9 when they sit at x0, x1 or x6, shrinks when they sit at x2, x5 or x7; weak overall fit."),
    4: ("A", "Fires when the last input x9 exceeds a running top value of x0..x8 (so x9 is the new maximum): about 10*ReLU(x9 - S - w*(M-S)), M,S = two largest of x0..x8, w about 0.7-1."),
    5: ("A", "Detects x8 standing out: about ReLU(12.9*x8 - 8*S7 - 2.6*mean7 + 1.1*M7 - 1.4), where M7,S7 = largest and second-largest of x0..x7 and mean7 their mean; large when x8 is the maximum or runner-up of x0..x8."),
    6: ("A", "Fires when x9 is below a running 'top value' of x0..x8: 10.7*ReLU(S + w*(M-S) - x9), M,S = two largest of x0..x8, w above 1 when M sits at x4-x5, about 0.3-0.4 when M sits at x0-x2 or x8."),
    7: ("A", "Fires when x9 is below the maximum of x0..x8: about 13*ReLU(M - x9) (with slight mixing toward the second-largest S when the max sits at x1-x4); M,S = two largest of x0..x8."),
    8: ("A", "Fires when x9 is below a running 'top value' of x0..x8: 12.3*ReLU(S + w*(M-S) - x9), M,S = two largest of x0..x8; w above 1 when M sits at x0-x2, near 0 when M sits at x5, x6 or x8."),
    9: ("A", "Small detector that x8 is the maximum of x0..x8: about ReLU(17.1*x8 - 15.6*M - 0.74*T7 - 0.23*x9 - 0.18), i.e. 1.5*x8 minus small terms when x8 = M; M = max of x0..x8, T7 = third-largest of x0..x7."),
    10: ("A", "Margin detector: (largest minus second-largest of x0..x8) weighted by the largest's position (strong at x0 and x5-x8, silent at x2-x3), plus (second minus third largest) weighted by second's position, plus third largest."),
    11: ("A", "Margin detector: (largest minus second-largest of x0..x8) weighted by the largest's position (strong at x0-x1, x7-x8, silent at x4), plus (second minus third largest) weighted similarly, plus third largest."),
    12: ("A", "Margin detector: (largest minus second-largest of x0..x8) weighted by the largest's position (strong at x0-x2, silent at x5), plus (second minus third largest) weighted similarly, plus third largest."),
    13: ("C", "Position-pattern code of the large values: grows with the maximum of x0..x9 (especially at x4-x5) and with runner-up values at x0, x1 or x5; shrinks when large values sit at x2, x6, x7 or x9."),
    14: ("A", "Margin detector: (largest minus second-largest of x0..x8) weighted by the largest's position (strong at x2-x4, silent at x7), plus (second minus third largest) and the third largest."),
    15: ("B", "Early-maximum detector: large when the maximum of x0..x9 sits at x0-x2 (scaled by its value) and when runner-up values sit at x0-x2 or x5; reduced when large values sit at x3-x9."),
}

CONSTANTS = {i: 5 for i in range(16)}

STRUCT = {
    0: "template Q2 (margin terms over sorted x0..x8); integer position codes u1,u2 in {-2..2} per position searched by coordinate descent",
    1: "template F+ (choice of features M,S of x0..x8, x9; cosine-in-position weight)",
    2: "template F2 (no x9 term)",
    3: "template R2 (top-3 of x0..x9); integer position codes u1,u2 in {-2..2} searched",
    4: "template F- (sign flipped)",
    5: "feature choice: x8, M7, S7, mean7 (4-subset search found nothing better)",
    6: "template F+",
    7: "template F+",
    8: "template F+",
    9: "feature choice: x8, x9, T7, M (searched all 4-subsets of 13 candidate order-statistic features)",
    10: "template Q2; integer position codes searched",
    11: "template Q2; integer position codes searched",
    12: "template Q2; integer position codes searched",
    13: "template R2; integer position codes searched",
    14: "template Q2; integer position codes searched",
    15: "template R2; integer position codes searched",
}

_F = {  # b, c0, A, omega, phi ; w(am) = c0 + A*cos(omega*am + phi)
    7: (13.0447, 0.8293, 0.2489, 0.6955, -11.1541, 1),
    1: (10.3404, 0.8475, -0.7182, 0.9293, 0.2153, 1),
    6: (10.7154, 0.7769, 0.5217, 0.8442, 2.3248, 1),
    8: (12.3218, 0.6335, 0.741, 0.5839, 5.6411, 1),
    4: (10.2413, 0.8294, -0.1741, 0.6682, 4.6661, -1),
    2: (18.1198, 0.8067, 0.2404, 0.6466, 2.1858, 0),
}
_Q = {  # u1 (by position of M), u2 (by position of S), consts a1,b1,a2,b2,g
    0: ([-1, 0, 1, 1, 1, 0, -1, -2, -1], [1, 0, 0, 0, 0, 0, 0, -2, 2], (14.698, 8.951, 5.8, 3.821, 2.49)),
    10: ([1, 0, -1, -2, 0, 1, 2, 2, 2], [2, 1, 0, -2, 0, 1, 1, 1, 2], (4.567, 9.753, 4.607, 4.431, 3.848)),
    11: ([1, 1, 0, -1, -2, -1, 0, 1, 1], [2, 1, 0, -1, -2, 0, 1, 1, 1], (10.465, 10.618, 6.664, 3.989, 3.754)),
    12: ([1, 1, 1, 0, -1, -2, -1, 0, 0], [2, 1, 0, -1, -1, -2, -1, 0, 1], (16.525, 13.58, 11.767, 5.038, 7.352)),
    14: ([-1, 0, 1, 2, 1, 0, -1, -2, 0], [-2, -1, 0, 0, 0, 0, 0, -1, 1], (12.585, 7.794, 11.465, 3.954, 6.358)),
}
_R = {  # u1 (position of top value), u2 (positions of 2nd/3rd), consts a1,b1,a2,b2,e
    3: ([1, 2, 0, 0, 0, -1, 1, -1, 0, -1], [1, 2, -1, 0, -1, -2, 1, -2, -1, -1], (2.353, 1.58, 2.079, 1.931, 0.195)),
    13: ([0, 0, -1, 0, 1, 2, -1, -1, 0, -2], [2, 2, -2, -1, 0, 2, -2, -1, 0, 0], (5.648, 1.281, 0.716, 1.35, 0.133)),
    15: ([1, 2, 2, 0, -1, -1, -2, -2, -1, -1], [1, 2, 2, -2, -1, 1, -1, -2, 0, 0], (8.021, 2.899, -0.048, 2.208, 0.386)),
}


def _relu(z):
    return np.maximum(z, 0.0)


def _top3(a):
    o = np.argsort(-a, axis=1)[:, :3]
    v = np.take_along_axis(a, o, axis=1)
    return o, v


def f(i, x):
    x = np.asarray(x, dtype=float)
    if i in _F:
        b, c0, A, om, ph, sg = _F[i]
        o, v = _top3(x[:, :9])
        w = c0 + A * np.cos(om * o[:, 0] + ph)
        z = v[:, 1] + w * (v[:, 0] - v[:, 1])
        if sg == 0:
            return b * z
        return b * _relu(sg * (z - x[:, 9]))
    if i in _Q:
        u1, u2, (a1, b1, a2, b2, g) = _Q[i]
        u1 = np.array(u1, float); u2 = np.array(u2, float)
        o, v = _top3(x[:, :9])
        g1 = v[:, 0] - v[:, 1]; g2 = v[:, 1] - v[:, 2]
        return _relu(g1 * (a1 + b1 * u1[o[:, 0]]) + g2 * (a2 + b2 * u2[o[:, 1]]) + g * v[:, 2])
    if i in _R:
        u1, u2, (a1, b1, a2, b2, e) = _R[i]
        u1 = np.array(u1, float); u2 = np.array(u2, float)
        o, v = _top3(x)
        z = v[:, 0] * (a1 + b1 * u1[o[:, 0]]) + a2 * (v[:, 1] + v[:, 2]) \
            + b2 * (v[:, 1] * u2[o[:, 1]] + v[:, 2] * u2[o[:, 2]]) + e
        return _relu(z)
    s7 = -np.sort(-x[:, :8], axis=1)
    M7, S7 = s7[:, 0], s7[:, 1]
    if i == 5:
        return _relu(12.947 * x[:, 8] + 1.1449 * M7 - 7.9921 * S7 - 2.5616 * x[:, :8].mean(1) - 1.4457)
    if i == 9:
        M8 = x[:, :9].max(1)
        return _relu(17.096 * x[:, 8] - 15.556 * M8 - 0.735 * s7[:, 2] - 0.226 * x[:, 9] - 0.183)
    raise ValueError(i)
