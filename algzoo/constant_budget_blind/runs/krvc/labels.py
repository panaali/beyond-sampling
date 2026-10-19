"""Neuron labels and formulas for h_10 of the second-largest-position RNN.
Each formula is relu(sum_k c_k * feature_k + c0) over interpretable input features (see _feat).
"""
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


SPEC = {0: (['r(S9-x9)n8', 'rmo[2,6]', 'A_rxmo5', 'A_rxmo1'], [1.57436, 1.13415, 1.4137, 0.85384, -0.00889]), 1: (['g12n8'], [4.48776, 0.28899]), 2: (['r(x3-to)', 'top2_3', 'am3', 'r(x0-mo)'], [2.07274, 1.52452, 0.82195, 1.0366, 0.09139]), 3: (['A_am', 'top2_8', 'pos_a29', 'A_P6'], [-0.12138, -2.55429, 0.2087, 0.69235, -0.24664]), 4: (['a29_even', 'A_P5', 'A_xso9', 'a28'], [0.82601, 0.26476, -0.16156, -0.46656, -0.27545]), 5: (['g12', 'x2-mo', 'g12n8', 'A_P5'], [5.21242, -0.48632, -2.74806, 0.76855, 0.07113]), 6: (['r(S9-x9)', 'g12n8', 'A_rxmo0', 'A_rxmo2'], [1.90647, 0.49923, 1.74877, 1.61184, -0.12232]), 7: (['r(x9-M9)'], [8.83872, 0.00305]), 8: (['A_P6', 'r(x5-to)', 'A_min', 'A_earl4'], [1.09086, 1.46497, -0.70773, -0.34832, 0.27027]), 9: (['S9', 'T9', 'x9', 'g12'], [-2.01741, -1.1687, -0.20811, -0.13069, 0.05631]), 10: (['amA4', 'amA5', 'pos_a29', 'r(M9-x9)'], [0.90089, 0.81194, -0.08857, 0.13318, -0.08849]), 11: (['am4*g12', 'am9_even', 'am8', 'am2'], [1.48212, 1.05802, -1.12576, -0.90271, 0.06801]), 12: (['r(S9-x9)'], [3.09775, 0.03808]), 13: (['r(S9-x9)n8', 'g12n8', 'amA7', 'A_rxmo2'], [1.75079, 0.58626, 0.86583, 1.13667, -0.05883]), 14: (['r(so-x8)', 'r(x0-to)', 'A_earl4', 'A_min'], [-5.21979, 1.10734, -0.27416, -0.47411, 0.95132]), 15: (['r(x1-to)', 'top2_1', 'am1', 'r(x1-lat)'], [1.5836, 1.20572, 0.96935, 1.70523, 0.19968]), 16: (['r(x9-S9)'], [5.68587, 0.00709]), 17: (['M9', 'S9'], [8.09693, 2.0234, 0.04545]), 18: (['r(S9-x9)', 'A_rxmo1', 'A_rxmo0', 'a22'], [1.39667, 2.76738, 1.008, 0.47408, -0.06369]), 19: (['r(x6-to)', 'top2_6', 'am6', 'r(x0-mo)'], [1.3301, 1.3455, 0.67267, 0.6424, 0.12672]), 20: (['x0-so', 'r(x1-so)', 'r(x8-so)', 'top2_0'], [-0.47556, 1.00329, 0.68061, -0.83669, 1.03138]), 21: (['A_earl9', 'A_rxso9', 'am9_even', 'A_rxso7'], [2.08013, 2.18627, 1.08899, -1.32853, -0.66008]), 22: (['r(x7-to)', 'top2_7', 'am7', 'A_rearl7'], [0.76735, 0.47858, 1.92935, 0.97668, 0.01912]), 23: (['x9-M9', 'r(x9-S9)', 'am8', 'A_P6'], [1.27386, 3.48688, -5.72952, 0.46145, -0.26106]), 24: (['x5-so', 'top2_5', 'A_P2_8', 'r(mo-x6)'], [-0.39685, -1.82051, 0.68522, 0.17725, 0.4319]), 25: (['x8-so', 'A_P2_7', 'top2_8', 'pos_a29'], [-0.23319, 0.62821, -0.98522, -0.07127, 0.69114]), 26: (['r(x2-to)', 'r(x0-so)', 'am4*g12', 'am2'], [2.83205, 2.09853, 2.17152, 2.0228, 0.63031]), 27: (['am[2,6]', 'rmo[2,6]'], [1.123, 1.26565, 0.02227]), 28: (['A_P6', 'am8', 'r(x0-to)', 'r(x1-to)'], [0.63367, -1.95996, 0.41584, 0.30773, 0.56033]), 29: (['r(S9-x9)n8', 'A_rxmo3', 'A_rxmo6', 'A_rxmo7'], [1.69968, 2.2131, 1.13976, 1.12158, 0.06632]), 30: (['x4-so', 'A_P9', 'top2_4', 'am3'], [-0.63635, 0.67678, -1.26249, 0.72735, 0.46671]), 31: (['S9', 'r(mo-x2)', 'A_P5', 'A_rxso2'], [4.83756, 0.3294, 1.06194, -0.87844, 0.37315])}

LABELS = {0: ('B', 'Measures how far x9 falls below the second-largest of x0..x8, switched off when x8 is the largest of x0..x8; also driven by the overall maximum being x1, x2, x5 or x6 (its margin).'), 1: ('A', 'Gap between the largest and second-largest of x0..x8, switched off when the largest of x0..x8 is x8 itself.'), 2: ('A', "Detects x3 being among the two largest of x0..x8: grows with x3's margin over the third-largest of the others, extra when x3 is the maximum; small offshoot when x0 is the maximum."), 3: ('C', 'Fires when the overall maximum is early and the second-largest of x0..x8 is late (grows with its position), scaled up by max(x0..x5); silenced when x8 is among the two largest of x0..x8.'), 4: ('C', 'Mostly a parity code: active when the second-largest of x0..x8 sits at an even position (not x8), plus weak dependence on max(x0..x4) and on x9 lying below the second-largest of x0..x8.'), 5: ('A', "Gap between the largest and second-largest of x0..x8, about twice as strong when x8 is the largest; offshoots: plus max of x0..x4, plus x2's shortfall below the maximum of the others."), 6: ('B', 'How far x9 falls below the second-largest of x0..x8, plus the top gap of x0..x8 (unless x8 is the largest), plus the margin by which x0 or x2 is the overall maximum.'), 7: ('A', 'How far x9 exceeds the largest of x0..x8; zero unless x9 is the overall maximum.'), 8: ('C', "Sum of several unrelated signals: max of x0..x5, x5's margin when it is among the two largest of x0..x8, minus the overall minimum, minus x4's excess over max(x0..x3)."), 9: ('A', 'Rarely active: fires only when the second- and third-largest of x0..x8 are both unusually low (roughly below zero); weaker when x9 or the top gap of x0..x8 is large.'), 10: ('A', 'Fires when the overall maximum of all ten values is x4 or x5, more so when the second-largest of x0..x8 comes early; small boost when x9 is below the maximum of x0..x8.'), 11: ('B', 'Codes the position of the maximum of x0..x8: on for even positions (x4 scaled by its gap over the second-largest), suppressed when the maximum is x2 or x8.'), 12: ('A', 'How far x9 falls below the second-largest of x0..x8; zero when x9 is among the two largest overall.'), 13: ('B', 'How far x9 falls below the second-largest of x0..x8, switched off when x8 is the largest of x0..x8; plus the top gap of x0..x8 and bonuses when the overall maximum is x7 or x2.'), 14: ('A', "Detects x8 being among the two largest of x0..x8: high when x8 is at least the second-largest of the others, dropping sharply as x8 falls below it; offshoots: x0's top-two margin, minus overall minimum."), 15: ('A', "Detects x1 being among the two largest of x0..x8: grows with x1's margin over the third-largest of the others and over all later values x2..x8, extra when x1 is the maximum."), 16: ('A', 'How far x9 exceeds the second-largest of x0..x8; zero unless x9 is among the two largest overall.'), 17: ('A', 'Running-maximum memory: about 8 times the largest plus 2 times the second-largest of x0..x8.'), 18: ('B', 'How far x9 falls below the second-largest of x0..x8, plus the margin by which x1 (strongly) or x0 is the overall maximum, plus a bonus when x2 is the second-largest of x0..x8.'), 19: ('A', "Detects x6 being among the two largest of x0..x8: grows with x6's margin over the third-largest of the others, extra when x6 is the maximum; small offshoot when x0 is the maximum."), 20: ('A', 'Anti-detector for x0: grows as x0 falls below the second-largest of the other values in x0..x8, near zero when x0 is among the two largest; offshoots for x1 or x8 being among the top two.'), 21: ('B', 'Fires when x9 is among the two largest overall (its excess over the second-largest of x0..x8, plus x9 minus their maximum); stronger if the maximum of x0..x8 is at an even position, weaker if x7 is top-two.'), 22: ('A', "Detects x7 being the maximum of x0..x8 (also, more weakly, the second-largest), growing with x7's margin over the third-largest of the others and over max(x0..x6)."), 23: ('B', 'Fires when x9 is among the two largest overall: how far x9 exceeds the second-largest of x0..x8 plus x9 minus their maximum; silenced when x8 is the largest of x0..x8; raised by max(x0..x5).'), 24: ('A', 'Anti-detector for x5: grows as x5 falls below the second-largest of the other values in x0..x8 (scaled with the second-largest of x0..x7), near zero when x5 is among the two largest of x0..x8.'), 25: ('A', 'Anti-detector for x8: grows as x8 falls below the second-largest of x0..x7 (and with the second-largest of x0..x6), near zero when x8 is among the two largest of x0..x8.'), 26: ('B', "Codes a maximum of x0..x8 at x2 (strongest: x2's margin over the third-largest), x0 (margin over second-largest) or x4 (gap over second-largest)."), 27: ('A', 'Fires when the maximum of x0..x8 is x2 or x6, growing with its margin over the next-largest value.'), 28: ('B', 'Max of x0..x5, strongly reduced when x8 is the largest of x0..x8; offshoots: margins of x0 and x1 over the third-largest of the other values of x0..x8.'), 29: ('B', 'How far x9 falls below the second-largest of x0..x8, switched off when x8 is the largest of x0..x8; plus the margin by which x3, x6 or x7 is the overall maximum.'), 30: ('A', 'Anti-detector for x4: grows as x4 falls below the second-largest of the other values in x0..x8 and with the maximum of x0..x8, near zero when x4 is among the two largest; bonus when x3 is the maximum.'), 31: ('A', "Mostly about 5 times the second-largest of x0..x8; offshoots: plus max(x0..x4), plus x2's shortfall below the maximum of the others, minus x2's margin when it is in the overall top two.")}

CONSTANTS = {0: 5, 1: 2, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 2, 8: 5, 9: 5, 10: 5, 11: 5, 12: 2, 13: 5, 14: 5, 15: 5, 16: 2, 17: 3, 18: 5, 19: 5, 20: 5, 21: 5, 22: 5, 23: 5, 24: 5, 25: 5, 26: 5, 27: 3, 28: 5, 29: 5, 30: 5, 31: 5}

STRUCT = {0: 'features chosen by greedy search over a library of order-statistic/position features: r(S9-x9)n8, rmo[2,6], A_rxmo5, A_rxmo1', 1: 'features chosen by greedy search over a library of order-statistic/position features: g12n8', 2: 'features chosen by greedy search over a library of order-statistic/position features: r(x3-to), top2_3, am3, r(x0-mo)', 3: 'features chosen by greedy search over a library of order-statistic/position features: A_am, top2_8, pos_a29, A_P6', 4: 'features chosen by greedy search over a library of order-statistic/position features: a29_even, A_P5, A_xso9, a28', 5: 'features chosen by greedy search over a library of order-statistic/position features: g12, x2-mo, g12n8, A_P5', 6: 'features chosen by greedy search over a library of order-statistic/position features: r(S9-x9), g12n8, A_rxmo0, A_rxmo2', 7: 'features chosen by greedy search over a library of order-statistic/position features: r(x9-M9)', 8: 'features chosen by greedy search over a library of order-statistic/position features: A_P6, r(x5-to), A_min, A_earl4', 9: 'features chosen by greedy search over a library of order-statistic/position features: S9, T9, x9, g12', 10: 'features chosen by greedy search over a library of order-statistic/position features: amA4, amA5, pos_a29, r(M9-x9)', 11: 'features chosen by greedy search over a library of order-statistic/position features: am4*g12, am9_even, am8, am2', 12: 'features chosen by greedy search over a library of order-statistic/position features: r(S9-x9)', 13: 'features chosen by greedy search over a library of order-statistic/position features: r(S9-x9)n8, g12n8, amA7, A_rxmo2', 14: 'features chosen by greedy search over a library of order-statistic/position features: r(so-x8), r(x0-to), A_earl4, A_min', 15: 'features chosen by greedy search over a library of order-statistic/position features: r(x1-to), top2_1, am1, r(x1-lat)', 16: 'features chosen by greedy search over a library of order-statistic/position features: r(x9-S9)', 17: 'features chosen by greedy search over a library of order-statistic/position features: M9, S9', 18: 'features chosen by greedy search over a library of order-statistic/position features: r(S9-x9), A_rxmo1, A_rxmo0, a22', 19: 'features chosen by greedy search over a library of order-statistic/position features: r(x6-to), top2_6, am6, r(x0-mo)', 20: 'features chosen by greedy search over a library of order-statistic/position features: x0-so, r(x1-so), r(x8-so), top2_0', 21: 'features chosen by greedy search over a library of order-statistic/position features: A_earl9, A_rxso9, am9_even, A_rxso7', 22: 'features chosen by greedy search over a library of order-statistic/position features: r(x7-to), top2_7, am7, A_rearl7', 23: 'features chosen by greedy search over a library of order-statistic/position features: x9-M9, r(x9-S9), am8, A_P6', 24: 'features chosen by greedy search over a library of order-statistic/position features: x5-so, top2_5, A_P2_8, r(mo-x6)', 25: 'features chosen by greedy search over a library of order-statistic/position features: x8-so, A_P2_7, top2_8, pos_a29', 26: 'features chosen by greedy search over a library of order-statistic/position features: r(x2-to), r(x0-so), am4*g12, am2', 27: 'features chosen by greedy search over a library of order-statistic/position features: am[2,6], rmo[2,6]', 28: 'features chosen by greedy search over a library of order-statistic/position features: A_P6, am8, r(x0-to), r(x1-to)', 29: 'features chosen by greedy search over a library of order-statistic/position features: r(S9-x9)n8, A_rxmo3, A_rxmo6, A_rxmo7', 30: 'features chosen by greedy search over a library of order-statistic/position features: x4-so, A_P9, top2_4, am3', 31: 'features chosen by greedy search over a library of order-statistic/position features: S9, r(mo-x2), A_P5, A_rxso2'}

def f(i, x):
    x = np.asarray(x, dtype=float)
    cols, c = SPEC[i]
    z = np.full(len(x), c[-1])
    for name, w in zip(cols, c[:-1]):
        z = z + w * _feat(name, x)
    return np.maximum(z, 0.0)
