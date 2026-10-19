"""Neuron labels and formulas for h_10 (16 neurons). Only numpy is used."""
import numpy as np
def _r(z): return np.maximum(z, 0.0)
def feature(name, x):
    """Interpretable features of x (N,10). s0=max, s1=second largest, am=argmax position,
    M_k=max(x0..x_k), S_k=second largest of x0..x_k, g=s0-s1."""
    x = np.asarray(x, float)
    s = -np.sort(-x, 1); s0, s1 = s[:, 0], s[:, 1]; am = np.argmax(x, 1); g = s0 - s1
    M = np.maximum.accumulate(x, 1)
    def S(k): return -np.sort(-x[:, :k+1], 1)[:, 1]
    def argM(k): return np.argmax(x[:, :k+1], 1)
    PO = ((am % 2 == 1) & (am < 8)).astype(float); PE = ((am % 2 == 0) & (am < 8)).astype(float)
    po = (am % 2 == 1).astype(float); pe = 1 - po
    n = name
    if n == 's0': return s0
    if n.startswith('max(x'): a=int(n[5]); b=int(n[8]); return np.maximum(x[:, a], x[:, b])
    if n.startswith('min(x'): a=int(n[5]); b=int(n[8]); return np.minimum(x[:, a], x[:, b])
    if n.startswith('cap'): t=int(n[3:]); return np.minimum(x[:, t], M[:, t-1])
    if n.startswith('r(x') and n[4]=='-' and n[5]=='x': return _r(x[:, int(n[3])] - x[:, int(n[6])])
    if n == 'g': return g
    if n.startswith('M9-x'): return s0 - x[:, int(n[4:])]
    if n.startswith('Mpo'): k = int(n[3:]); return (argM(k) % 2 == 1) * M[:, k]
    if n.startswith('Mpe'): k = int(n[3:]); return (argM(k) % 2 == 0) * M[:, k]
    if n == 'E8M8': a = argM(8); return ((a % 2 == 0) & (a < 8)) * M[:, 8]
    if n == 'PO*g': return PO * g
    if n == 'PE*g': return PE * g
    if n == 'pe*g': return pe * g
    if n == 'po*g': return po * g
    if n == 'pe*s0': return pe * s0
    if n == 'pe*s1': return pe * s1
    if n == 'PE*s0': return PE * s0
    if n == 'PO*s0': return PO * s0
    if n.startswith('po*(M9-x'): return po * (s0 - x[:, int(n[8])])
    if n.startswith('pe*x'): return pe * x[:, int(n[4:])]
    if n.startswith('am6*x'): return (am == 6) * x[:, int(n[5:])]
    if n == 'am6*s1': return (am == 6) * s1
    if n.startswith('r(x') and n.endswith('-m2)'): return _r(x[:, int(n[3])] - s1)
    if n.startswith('r(m2-x'): return _r(s1 - x[:, int(n[6])])
    if n.startswith('r(-x'): return _r(-x[:, int(n[4])])
    if n.startswith('r(x') and '-M' in n:  # r(x_t - M_k)
        t = int(n[3]); k = int(n[n.index('-M')+2:-1]); return _r(x[:, t] - M[:, k])
    if n.startswith('r(x') and '-S' in n:
        t = int(n[3]); k = int(n[n.index('-S')+2:-1]); return _r(x[:, t] - S(k))
    if n.startswith('r(x'): return _r(x[:, int(n[3])])
    if n.startswith('xn'): t = int(n[2:]); return x[:, t] * (am != t)
    if n.startswith('M') and '-x' in n: k = int(n[1:n.index('-')]); t = int(n[n.index('-x')+2:]); return M[:, k] - x[:, t]
    if n.startswith('M') and '-S' in n: k = int(n[1:n.index('-')]); return M[:, k] - S(k)
    if n.startswith('M'): return M[:, int(n[1:])]
    if n.startswith('S'): return S(int(n[1:]))
    if n.startswith('x'): return x[:, int(n[1:])]
    raise KeyError(n)


LABELS = {
    0: ('A', 'Memory of the running maximum of x0..x8 (strong positive gain), minus the recent inputs x7 and x8, plus small terms for how far x9 and x4 fall below the maximum.'),
    1: ('B', 'How far x6 lies below the overall maximum (max minus x6), plus a top-two-gap term (largest minus second largest) that is positive when the maximum sits at odd x1/x3/x5/x7 and negative otherwise.'),
    2: ('A', 'How far x9 lies below the overall maximum (max minus x9), corrected by the gap between largest and second largest: gap subtracted when the maximum is at odd x1..x7, added when at even x0..x6. ReLU.'),
    3: ('B', 'Running maximum of x0..x7, tilted by early positions: plus how much x2 exceeds x4 and max(x0,x2), minus max(x1,x3) and max(x1,x5) (even positions add, odd subtract).'),
    4: ('B', 'Running maximum of x0..x7 with large gain, lowered when the max of x0..x6 is at an even position or the top-two gap is large with even-positioned max; minus min(x4,x6) and max(x0,x6).'),
    5: ('A', 'Memory of the running maximum of x0..x8, plus how far x8 falls below the overall maximum, minus how far x9 falls below it, minus min(x4,x7) and min(x5,x6).'),
    6: ('A', "How far x7 lies below the running maximum of x0..x7, i.e. max(x0..x7) minus x7, with small corrections for the top-two gap and for the parity of the earlier maximum's position."),
    7: ('A', 'Memory of the running maximum of x0..x8, boosted when that maximum sits at an even position x0..x6; minus x5 and x7 when they are not the maximum; small plus for x4.'),
    8: ('A', 'Running maximum of x0..x8, minus x8 and x5 each capped at the running maximum before them, minus the top-two gap when the maximum is at even x0..x6, minus how far x9 falls below the second largest.'),
    9: ('A', 'Running maximum of x0..x7 minus a weighted sum of mid-sequence inputs: x4 and x5 most strongly, then the larger of x2,x6 and the larger of x0,x2.'),
    10: ('A', 'Detects x9 being the largest value: equals a gain times (x9 minus the second-largest value) when x9 is the maximum, zero otherwise.'),
    11: ('A', 'How far x9 lies below the overall maximum (max minus x9), minus the gap between largest and second-largest values when the maximum sits at an even position (x0,x2,..,x8). ReLU.'),
    12: ('B', 'Tracks x3: how much x3 exceeds x1, plus x3 capped at max(x0,x1,x2), plus max(x0,x2) and how much x4 exceeds x5, minus positive x1. ReLU; active about half the time.'),
    13: ('A', 'Running maximum of x0..x8, strongly boosted by how much x8 exceeds max(x0..x7) (a new maximum at x8); minus max(x2,x4) and min(x5,x7), plus how much x0 exceeds x6.'),
    14: ('A', "Detects x6 being the largest value: when it is, roughly proportional to x6, reduced by x9, x3, x7 and by x6's lead over the second-largest value; zero otherwise."),
    15: ('C', 'Mixed: grows with how far x5 lies below max(x0..x5), with positive x2, with x6 exceeding max(x0..x5), with x1 exceeding x0, and with the top-two gap when the maximum is at even x0..x6.'),
}

FORMULAS = {
    0: (['M8', 'x8', 'x7', 'M9-x9', 'M4-x4'], [48.443, -14.62, -9.99, 9.21, 6.398]),
    1: (['M9-x6', 'Mpo8', 'xn6', 'PO*g', 'g'], [10.673, 2.993, -1.357, 15.692, -9.777]),
    2: (['M9-x9', 'PO*g', 'PE*g'], [12.808, -11.902, 7.219]),
    3: (['M7', 'r(x2-x4)', 'max(x1,x3)', 'max(x1,x5)', 'max(x0,x2)'], [19.788, 6.995, -5.103, -4.435, 3.295]),
    4: (['M7', 'Mpe6', 'min(x4,x6)', 'max(x0,x6)', 'PE*g'], [37.284, -10.934, -5.598, -6.302, -11.712]),
    5: (['M8', 'min(x4,x7)', 'M9-x8', 'M9-x9', 'min(x5,x6)'], [36.402, -10.958, 7.419, -5.566, -5.298]),
    6: (['M7-x7', 'Mpo6', 'PE*g'], [13.009, -2.229, 2.123]),
    7: (['x4', 'M8', 'xn5', 'xn7', 'E8M8'], [3.181, 35.045, -4.071, -4.708, 8.51]),
    8: (['M8', 'cap8', 'PE*g', 'cap5', 'r(m2-x9)'], [33.764, -10.817, -17.611, -5.607, -2.585]),
    9: (['M7', 'min(x4,x5)', 'max(x2,x6)', 'max(x4,x5)', 'max(x0,x2)'], [20.921, -6.219, -4.531, -4.744, -3.102]),
    10: (['r(x9-m2)'], [15.991]),
    11: (['M9-x9', 'pe*g'], [11.404, -9.23]),
    12: (['r(x3-x1)', 'cap3', 'max(x0,x2)', 'r(x4-x5)', 'r(x1)'], [5.191, 8.76, 2.094, 1.991, -2.122]),
    13: (['M8', 'r(x8-M7)', 'max(x2,x4)', 'r(x0-x6)', 'min(x5,x7)'], [18.88, 32.866, -4.602, 3.084, -2.496]),
    14: (['am6*x6', 'am6*x9', 'r(x6-m2)', 'am6*x3', 'am6*x7'], [3.462, -1.454, -1.64, -0.635, -0.464]),
    15: (['M5-x5', 'r(x2)', 'r(x6-M5)', 'r(x1-M0)', 'PE*g'], [6.547, 13.432, 8.726, 3.901, 7.146]),
}

CONSTANTS = {i: len(FORMULAS[i][1]) for i in range(16)}

STRUCT = {i: "feature set chosen by greedy/beam search over a dictionary of max/rank/position features "
             "(running maxima M_k, capped values, pairwise max/min, ReLU differences, argmax-parity indicators); "
             "features: " + ", ".join(FORMULAS[i][0]) + "; output passed through ReLU" for i in range(16)}

def f(i, x):
    """Predict h_10[i]: ReLU of a weighted sum of named features (see feature())."""
    x = np.asarray(x, float)
    names, w = FORMULAS[i]
    z = sum(wk * feature(n, x) for n, wk in zip(names, w))
    return np.maximum(z, 0.0)
