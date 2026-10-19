"""Neuron labels and formulas for h_10 of the second-largest RNN.
Every formula is f = ReLU(sum_k c_k * feature_k) over interpretable features (see _feat)."""
import numpy as np

LABELS = {0: ('A', 'Fires when an earlier value is high but x9 is low: ReLU(~13*(max of x0..x8, taken as 0 if x8 holds it) - 7.6*x9), weaker when that max sits at x1,x4,x5,x7, strongest at x0,x2,x3,x6.'), 1: ('B', 'Large when the top of x0..x7 sits at x2, x5 or x6 and exceeds the rest: ReLU(25*max(x0..x7) - ~4-6 times each of max(x0,x1,x3,x4,x7), max(x0,x4,x7), max(x0..x8 except x2,x5), second largest of x0..x8 except x2).'), 2: ('D', 'Weak, poorly understood: roughly ReLU(8*(second largest of x0,x1,x2,x3,x4,x6,x7) + 4*x2 - 4*(second largest of x0..x8 unless at x2) + small terms in max(x1,x4) and x2 when third largest).'), 3: ('A', 'Fires when an earlier value is high but x9 is low: ReLU(~12.7*(max of x0..x8, taken as 0 if x8 holds it) - 8*x9), weaker when that max sits at x0,x3,x6,x7, strongest at x1,x2,x4,x5.'), 4: ('A', 'Margin of the largest over the second largest of x0..x8: ReLU(~16.6*largest - ~16*second + 5.4*max(x0,x1,x3,x4,x6,x7)); zero when x8 is the largest, smaller when the second largest is x8.'), 5: ('C', 'Mostly ~8*max(x0,x2,x3,x5,x6,x7,x8), plus ~4*largest of x0..x8 when it is at x0,x2,x3,x5,x6,x8; strongly reduced when the largest or second largest of x0..x8 is x5. ReLU applied.'), 6: ('A', 'Fires when an earlier value is high but x9 is low: ReLU(~12.4*(max of x0..x8, taken as 0 if x8 holds it) - 8*x9), weaker when that max sits at x2,x3,x5,x6, strongest at x0,x1,x4,x7.'), 7: ('A', 'Stores the running maximum: ~17*max(x0..x7) + 5*max(x0,x3,x6) + 7*max(x0,x2,x3,x5,x6,x7,x8) + 2.6*(max of x0..x8 if at x3 or x6); strongest when the maximum sits at x0,x3,x6.'), 8: ('B', 'ReLU(13*max(x0,x1,x3,x4,x7) - 7*second largest of x0..x7) plus bonuses (~2-4 times the largest of x0..x8) when that largest sits at x4 or x7; bonuses missing when it sits at x2 or x3.'), 9: ('C', 'ReLU(6.4*max(x0,x1,x4,x7) + 3*x1 + ~3.4*largest of x0..x8 if at x0,x1,x3,x4,x7 + ~1.7*that largest if not at x0,x3,x6 + 3.8*second largest of x0..x8 if it is x1).'), 10: ('A', 'Last input beats the earlier runner-up: ReLU(8.4*x9 - 4.7*P2 - 1.9*P3), P2,P3 = second and third largest of x0..x8. Positive essentially when x9 is among the two largest inputs.'), 11: ('C', 'ReLU(8*(largest of x0..x8 if at x1,x2,x3,x5,x6,x8) + 5.2*(that largest if at x0,x2,x4,x5,x8) - 5.3*(second largest of x0..x8 unless at x8) - 5.1*(second largest if at x7)).'), 12: ('A', 'Margin of the largest over the second largest of x0..x8: ReLU(21*largest - 12.4*second - 2.3*max(x2,x5) - 3*max(x1,x2,x5,x6,x8)); smaller when the big values sit at x2,x5.'), 13: ('D', 'Partly understood: ReLU(10.4*max(x0,x1,x3,x4,x7) + 7.6*second largest of x0,x1,x3,x4,x7 - 2.6*max(x4,x7) + 1.4*x2 - 2.8*(second largest of x0..x8 if at x4 or x7)).'), 14: ('A', 'Maximum read with a position-dependent weight: ReLU(10*max(x1,x2,x4,x5,x7,x8) + 15.5*max(x0,x1,x2,x3,x5,x6,x8) - 5.3*max(x0,x2,x3,x6) + 4.5*(max of x0..x8 if at x1,x2,x4,x5,x7,x8)).'), 15: ('A', 'New maximum at the very end: 26.8*ReLU(x9 - 0.989*max(x0..x8)); positive only when x9 is the largest of all ten inputs, growing with its lead.')}

SPEC = {0: ['x9', 'P1@01234567', 'P1@14578', 'P1@01234678', 'max01234578'], 1: ['M7', 'max01347', 'max047', 'max0134678', '2nd01345678'], 2: ['2nd0123467', 'x2', 'P2@01345678', 'max14', 'P3@2'], 3: ['x9', 'P1@01234567', 'P1@0367', 'max01234678', 'P2'], 4: ['x9', 'P1@01234567', 'P2', 'max013467', 'P2@01234567'], 5: ['max0235678', 'P1@5', 'P1@023568', 'P1@01345678', 'P2@5'], 6: ['x9', 'P1@01234567', 'P1@23568', 'max01235678', 'P2'], 7: ['x9', 'M7', 'max036', 'max0235678', 'P1@36'], 8: ['max01347', 'P1@47', 'P1@01345678', '2nd01234567', 'P1@01245678'], 9: ['max0147', 'x1', 'P1@01347', 'P1@124578', 'P2@1'], 10: ['x9', 'P2', 'P3'], 11: ['x9', 'P1@123568', 'P1@02458', 'P2@01234567', 'P2@7'], 12: ['x9', 'P1', 'P2', 'max25', 'max12568'], 13: ['max01347', '2nd01347', 'max47', 'x2', 'P2@47'], 14: ['max124578', 'max0123568', 'max0236', 'P1@124578'], 15: ['x9', 'P1']}

COEF = {0: [-7.607, 12.935, -3.122, -1.869, -3.027], 1: [25.125, -3.781, -4.946, -5.808, -4.12], 2: [7.974, 3.976, -3.985, 1.887, 3.541], 3: [-7.962, 12.653, -2.835, -6.807, 3.474], 4: [0.267, 16.578, -21.175, 5.43, 5.231], 5: [8.252, -7.352, 3.98, 2.527, -4.163], 6: [-8.093, 12.443, -3.138, -5.217, 2.487], 7: [0.171, 17.424, 5.147, 7.278, 2.618], 8: [13.406, 3.699, 3.427, -7.009, 2.337], 9: [6.422, 3.006, 3.354, 1.659, 3.825], 10: [8.392, -4.693, -1.946], 11: [0.229, 7.96, 5.23, -5.347, -5.138], 12: [-0.206, 21.238, -12.386, -2.286, -3.008], 13: [10.412, 7.572, -2.577, 1.395, -2.81], 14: [10.071, 15.537, -5.278, 4.549], 15: [26.803, -26.505]}

CONSTANTS = {i: len(COEF[i]) for i in range(16)}

STRUCT = {i: "searched: feature list " + ", ".join(SPEC[i]) + " (position sets chosen by greedy search, mostly arcs of the phase circle 2,5,8,1,4,7,0,3,6)" for i in range(16)}

def _feat(name, x):
    """Interpretable features of x (N,10). P1>=P2>=P3 are the sorted values of the prefix x0..x8;
    pa, pb, pc are their positions. 'maxS' = max over positions in S; '2ndS' = second largest over S;
    'P1@S' = P1 if its position pa is in S else 0 (same for P2@S with pb, P3@S with pc)."""
    pre = x[:, :9]
    o = np.argsort(-pre, axis=1)
    srt = np.take_along_axis(pre, o, axis=1)
    if name in ('P1', 'P2', 'P3', 'P4'):
        return srt[:, int(name[1]) - 1]
    if name == 'M7':
        return x[:, :8].max(axis=1)
    if name[0] == 'x':
        return x[:, int(name[1:])]
    if name.startswith('max'):
        S = [int(c) for c in name[3:]]
        return x[:, S].max(axis=1)
    if name.startswith('2nd'):
        S = [int(c) for c in name[3:]]
        return -np.sort(-x[:, S], axis=1)[:, 1]
    if name[0] == 'P' and '@' in name:
        r = int(name[1]) - 1
        S = [int(c) for c in name.split('@')[1]]
        return srt[:, r] * np.isin(o[:, r], S)
    raise ValueError(name)


def f(i, x):
    x = np.asarray(x, dtype=float)
    z = np.zeros(len(x))
    for name, c in zip(SPEC[i], COEF[i]):
        z = z + c * _feat(name, x)
    return np.maximum(z, 0.0)
