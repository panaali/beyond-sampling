"""Labels for zoo_2nd_argmax(4, 3) hidden neurons (final hidden state).

Structure: two mirror-image pairs. Neurons 0,1 are live only when the LAST input
beats the MIDDLE one (x2 > x1); neurons 2,3 are the same units with the sign of x flipped
(live when x1 > x2). Neuron 3 is approximately neuron 0 applied to -x, and neuron 2 is approximately neuron 1 applied to -x.
Each neuron = ReLU( a*x2 + b*x1 + x0*(c + d*s) ), where s in [-1,1] is a soft "is x1 above x0?"
comparison: s = clip((x1-x0) / (w*|x0|), -1, 1), i.e. +1 if x1 is clearly above x0, -1 if clearly below,
linear in a narrow band around x1 == x0 (that band is the step-1 hidden pair switching sign).
"""
import numpy as np


def _unit(x, a, b, c, d, w):
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    s = np.clip((x1 - x0) / (w * np.abs(x0) + 1e-9), -1.0, 1.0)  # soft sign(x1 - x0)
    return np.maximum(a * x2 + b * x1 + c * x0 + d * x0 * s, 0.0)


LABELS = {
    0: ("A", "Fires when the last input x2 beats the middle input x1: ReLU(3.95*x2 - 4.9*x1 + k*x0), where k = +1.5 if x1 > x0 "
             "and k = +0.5 if x1 < x0 (soft switch near x1 = x0). x0 has a positive weight. Paired with neuron 1, the ratio "
             "n0/n1 places x0 relative to x1 and x2 (a large ratio means x0 is the largest). Mirror of neuron 3."),
    1: ("A", "Fires when the last input x2 beats the middle input x1: ReLU(4.7*x2 - 3.7*x1 - k*x0), where k = 0.5 if x1 > x0 "
             "and k = 1.3 if x1 < x0 (soft switch near x1 = x0). x0 has a negative weight. Paired with neuron 0, the ratio "
             "n0/n1 places x0 relative to x1 and x2. Mirror of neuron 2."),
    2: ("A", "Sign-flipped copy of neuron 1. Fires when the middle input x1 beats the last input x2: "
             "ReLU(-4.7*x2 + 3.7*x1 + k*x0), where k = 0.45 if x1 < x0 and k = 1.35 if x1 > x0 (soft switch). "
             "x0 has a positive weight."),
    3: ("A", "Sign-flipped copy of neuron 0. Fires when the middle input x1 beats the last input x2: "
             "ReLU(-3.95*x2 + 4.9*x1 - k*x0), where k = 1.5 if x1 < x0 and k = 0.5 if x1 > x0 (soft switch). "
             "x0 has a negative weight. With neuron 2, the ratio n3/n2 places x0 relative to x1 and x2."),
}

# (a, b, c, d, w), applied to sign*x: sign = +1 for neurons 0,1 and -1 for neurons 2,3
_P = {
    0: (3.951, -4.898, 1.022, 0.511, 0.230),
    1: (4.678, -3.690, -0.900, 0.414, 0.193),
    2: (4.699, -3.712, -0.892, 0.444, 0.244),
    3: (3.955, -4.896, 1.017, 0.531, 0.243),
}


def f(i, x):
    x = np.asarray(x, dtype=float)
    sign = 1.0 if i < 2 else -1.0
    return _unit(sign * x, *_P[i])
