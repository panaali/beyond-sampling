"""Per-neuron labels and formulas for h_10 of the second-argmax RNN.
Feature names: xj = input j; a1/a2/a3 = 1st/2nd/3rd largest of x0..x8; a1@{S} = a1 if its position is in S;
suffix g = multiplied by [largest of x0..x8 is not at x8]; s2@j = overall 2nd largest if at j;
Pt/Rt/Tt = largest/2nd/3rd of x0..xt; Mc = max(xc, xc+3, ...); max{S}/sec{S} = largest/2nd largest over positions S;
newj = max(0, xj - max(x0..x(j-1))); up9 = max(0, x9 - a1). Each formula: h = max(0, sum_k c_k * feature_k)."""
import numpy as np

import numpy as np
def _feature(x, name, cache=None):
    """Compute a named interpretable feature of x (N,10)."""
    N = len(x)
    o = np.argsort(-x, 1); s = np.take_along_axis(x, o, 1)          # overall ranks
    o9 = np.argsort(-x[:, :9], 1); a = np.take_along_axis(x[:, :9], o9, 1)  # ranks within x0..x8
    p = o9[:, 0]; g8 = (p != 8).astype(float)
    def topk(t, k):
        return -np.sort(-x[:, :t + 1], 1)[:, k]
    if name.startswith('max{') or name.startswith('sec{'):
        S = [int(v) for v in name[4:].strip('{}').split(',')]
        srt = np.sort(x[:, S], 1)
        return srt[:, -1] if name.startswith('max') else srt[:, -2]
    if '@' in name:
        base, sel = name.split('@')
        S = [int(v) for v in sel.strip('{}').split(',')]
        gate = base.endswith('g')
        if gate: base = base[:-1]
        k = int(base[1:]) - 1
        if base[0] == 's':
            return s[:, k] * np.isin(o[:, k], S)
        if base[0] == 'a':
            return a[:, k] * np.isin(o9[:, k], S) * (g8 if gate else 1.0)
    if name in ('s1', 's2', 's3'): return s[:, int(name[1]) - 1].copy()
    if name in ('a1', 'a2', 'a3'): return a[:, int(name[1]) - 1].copy()
    if name in ('a1g', 'a2g', 'x9g'):
        v = {'a1g': a[:, 0], 'a2g': a[:, 1], 'x9g': x[:, 9]}[name]; return v * g8
    if name[0] == 'x': return x[:, int(name[1:])].copy()
    if name == 'up9': return np.maximum(x[:, 9] - a[:, 0], 0)
    if name[0] == 'P': return topk(int(name[1:]), 0)
    if name[0] == 'R': return topk(int(name[1:]), 1)
    if name[0] == 'T': return topk(int(name[1:]), 2)
    if name[0] == 'Q': return x[:, int(name[1:]):].max(1)
    if name[0] == 'M': return x[:, int(name[1:])::3].max(1)
    if name.startswith('new'):
        j = int(name[3:]); return np.maximum(x[:, j] - x[:, :j].max(1), 0)
    raise ValueError(name)



LABELS = {0: ('A', 'Fires when x9 falls well below the largest of x0..x8: grows with that maximum (weight about 2x larger if it sits at x0,x2,x3,x5,x6 than at x1,x4,x7) and the second-largest, minus about 7.7*x9; silent if x8 is largest of x0..x8.'), 1: ('C', 'Mixture of maxima over position groups: max(x1,x2,x3,x5,x6) + max(x0,x2..x7) + max(x0..x8) minus max(x0,x3,x4,x6,x7,x8) and second-largest of x0,x1,x3..x7; strongest when the overall maximum is at x2 or x5.'), 2: ('D', 'Not well understood. Roughly the second-largest of x0..x4 plus a little of x2 itself; strongest when x2 is the largest or second-largest input, small otherwise.'), 3: ('A', 'Fires when x9 falls well below the largest of x0..x8: grows with that maximum (weight about 2x larger if it sits at x1,x2,x4,x5 than at x0,x3,x6,x7) and the second-largest, minus about 8.2*x9; silent if x8 is largest of x0..x8.'), 4: ('A', 'Margin of the maximum: largest of x0..x7 minus the second-largest of x0..x8 (so x8 being largest suppresses it), slightly lowered by large values at x2,x5,x8 and by x8 being second-largest overall.'), 5: ('A', 'Position-weighted copy of the maximum of x0..x8: large when the largest value is at x0,x2,x3,x6,x7 or x8, weak when at x1,x4,x5; lowered by the second-largest of x0,x1,x3,x4,x5,x6; x9 ignored.'), 6: ('A', 'Fires when x9 falls well below the largest of x0..x8: grows with that maximum (weight about 2x larger if it sits at x0,x1,x4,x7 than at x2,x3,x5,x6) and the second-largest, minus about 8.2*x9; silent if x8 is largest of x0..x8.'), 7: ('A', 'Size of the running maximum: mainly the largest of x0..x7, with extra weight when the largest value sits at x0,x3,x6 (also x2,x7,x8); almost always positive and ignores x9.'), 8: ('A', 'Position-weighted maximum: mainly max(x0,x1,x3,x4,x7), minus the second-largest of x0..x5, plus max(x0,x1,x4,x5,x6,x7) corrections; small when the largest value sits at x2, x5 or x9.'), 9: ('B', 'Largest value among x0,x1,x4,x7 plus extra direct weight on x1 and on x4 when x4 sets a new running record; reduced by the second-largest of x0,x4,x5,x6,x7. Small when the maximum is at x2,x3,x5,x6,x8,x9.'), 10: ('A', 'Signals that the last input x9 is among the top two: about 8.3*x9 minus 4.6*(second-largest of x0..x8) minus 2.0*(third-largest of x0..x8), floored at zero.'), 11: ('C', 'Largest of x2,x5,x6,x8 plus largest of x1..x6,x8, minus the second-largest of x0..x7 and minus maxima over x4,x6,x7 and x1,x3,x4,x7; essentially silent whenever x7 is the largest input.'), 12: ('A', 'Margin of the maximum: largest of x0..x8 minus the second- and third-largest of x0..x8, slightly lowered by large values at x1,x2,x5,x6,x8; x9 ignored.'), 13: ('B', 'Size of early large values: largest of x0,x1,x3,x4,x7 plus the second-largest of x0..x3 and of x0..x4, minus the second-largest of x0,x2,x4,x6; favours maxima at x0,x1,x3,x7.'), 14: ('A', 'Position-weighted copy of the maximum of x0..x8: large when the largest value sits at x1,x2,x4,x5,x7 or x8, weaker when at x0,x3,x6; x9 ignored.'), 15: ('A', 'How much the last input x9 exceeds every earlier input: about 27*max(0, x9 - max(x0..x8)); zero unless x9 is the overall maximum.')}
FEATURES = {0: ['a1g@{0,2,3,5,6}', 'a1@{1,4,7}', 'a2g', 'x9g'], 1: ['max{1,2,3,5,6}', 'sec{0,1,3,4,5,6,7}', 'max{0,2,3,4,5,6,7}', 'max{0,3,4,6,7,8}', 'P8'], 2: ['R4', 'x2', 'sec{0,1,3,5,8}', 'max{1,2}', 'sec{0,2,3,4}'], 3: ['a1g@{1,2,4,5}', 'a1@{0,3,6,7}', 'a2g', 'x9g'], 4: ['P7', 'R8', 'M2', 's2@8', 'max{1,2,4,5,6,8}'], 5: ['max{0,2,3,6,7,8}', 'max{0,3,6,8}', 'sec{0,1,3,4,5,6}', 'max{0,1,2,3,4,5,6,8}', 'max{1,4,5}'], 6: ['a1g@{0,1,4,7}', 'a1@{2,3,5,6}', 'a2g', 'x9g'], 7: ['P7', 'max{0,3,6,8}', 'max{0,2,3,6,7}', 'max{3,4,5,6}', 'max{0,1,2,3,6,8}'], 8: ['max{0,1,3,4,7}', 'R5', 'max{0,1,4,5,6,7}', 'max{1,2,5,6}', 'max{1,2,4,5,6,7}'], 9: ['max{0,1,4,7}', 'x1', 'sec{0,4,5,6,7}', 'max{1,3,4,5,7}', 'new4'], 10: ['x9', 'a2', 'a3'], 11: ['max{2,5,6,8}', 'max{4,6,7}', 'max{1,2,3,4,5,6,8}', 'R7', 'max{1,3,4,7}'], 12: ['P8', 'R8', 'max{1,2,5}', 'max{2,5,6,8}', 'T8'], 13: ['max{0,1,3,4,7}', 'R3', 'sec{0,2,4,6}', 'max{0,3,4,6,7}', 'R4'], 14: ['max{1,2,4,5,7,8}', 'max{0,1,2,3,5,6,8}', 'max{0,1,2,3,6}', 'max{0,3,5,6}', 'max{0,1,2,3,4,5,6,8}'], 15: ['up9']}
COEFS = {0: [6.436, 3.5514, 2.5094, -7.6633], 1: [7.3637, -9.3836, 9.8119, -8.4413, 8.0647], 2: [6.3747, 1.2603, -2.7783, 2.8528, 2.3984], 3: [6.6625, 3.1278, 3.2527, -8.2106], 4: [23.2858, -13.0292, -2.7733, -4.0015, -2.9551], 5: [6.7302, 2.9354, -2.9614, 6.8768, -3.1335], 6: [7.7705, 4.2591, 2.2991, -8.2299], 7: [13.1387, 3.0035, 7.3402, 5.2226, 3.3467], 8: [12.0569, -6.156, 6.0026, -6.2043, 5.6754], 9: [9.212, 2.9153, -3.9919, 3.0804, 3.3159], 10: [8.3475, -4.5743, -1.9931], 11: [2.9963, -3.1885, 11.6447, -5.2319, -4.0758], 12: [21.1231, -9.9636, -2.8019, -2.6199, -3.1798], 13: [6.6207, 6.1615, -3.3462, 3.1845, 3.4529], 14: [12.3777, 15.3987, -9.0415, -4.9517, 8.4189], 15: [27.0659]}
CONSTANTS = {0: 4, 1: 5, 2: 5, 3: 4, 4: 5, 5: 5, 6: 4, 7: 5, 8: 5, 9: 5, 10: 3, 11: 5, 12: 5, 13: 5, 14: 5, 15: 1}
STRUCT = {0: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: a1g@{0,2,3,5,6}; a1@{1,4,7}; a2g; x9g', 1: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: max{1,2,3,5,6}; sec{0,1,3,4,5,6,7}; max{0,2,3,4,5,6,7}; max{0,3,4,6,7,8}; P8', 2: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: R4; x2; sec{0,1,3,5,8}; max{1,2}; sec{0,2,3,4}', 3: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: a1g@{1,2,4,5}; a1@{0,3,6,7}; a2g; x9g', 4: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: P7; R8; M2; s2@8; max{1,2,4,5,6,8}', 5: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: max{0,2,3,6,7,8}; max{0,3,6,8}; sec{0,1,3,4,5,6}; max{0,1,2,3,4,5,6,8}; max{1,4,5}', 6: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: a1g@{0,1,4,7}; a1@{2,3,5,6}; a2g; x9g', 7: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: P7; max{0,3,6,8}; max{0,2,3,6,7}; max{3,4,5,6}; max{0,1,2,3,6,8}', 8: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: max{0,1,3,4,7}; R5; max{0,1,4,5,6,7}; max{1,2,5,6}; max{1,2,4,5,6,7}', 9: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: max{0,1,4,7}; x1; sec{0,4,5,6,7}; max{1,3,4,5,7}; new4', 10: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: x9; a2; a3', 11: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: max{2,5,6,8}; max{4,6,7}; max{1,2,3,4,5,6,8}; R7; max{1,3,4,7}', 12: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: P8; R8; max{1,2,5}; max{2,5,6,8}; T8', 13: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: max{0,1,3,4,7}; R3; sec{0,2,4,6}; max{0,3,4,6,7}; R4', 14: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: max{1,2,4,5,7,8}; max{0,1,2,3,5,6,8}; max{0,1,2,3,6}; max{0,3,5,6}; max{0,1,2,3,4,5,6,8}', 15: 'features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: up9'}

def f(i, x):
    x = np.asarray(x, dtype=float)
    F = np.stack([_feature(x, n) for n in FEATURES[i]], 1)
    return np.maximum(F @ np.array(COEFS[i]), 0.0)
