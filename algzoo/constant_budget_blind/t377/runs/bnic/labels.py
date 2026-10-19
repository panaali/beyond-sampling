import numpy as np

def _feats(x):
    x = np.asarray(x, dtype=float)
    M = np.maximum.accumulate(x, 1)          # running maxima M_k = max(x0..xk)
    m = M[:, 9]                               # overall max
    am = np.argmax(x, 1)                      # position of the max
    T = m[:, None] - x                        # T_j = gap of x_j below the overall max
    F = {}
    F['M9'] = m
    F['G9'] = np.maximum(x[:, 9] - M[:, 8], 0)   # amount by which x9 is a new record
    F['P9'] = np.maximum(M[:, 8] - x[:, 9], 0)   # amount by which x9 falls below max(x0..x8)
    F['negsum'] = np.maximum(-x, 0).sum(1)       # total size of the negative inputs
    ev = np.isin(am, [0, 2, 4, 6, 8]) * 1.0
    F['M9_amEven'] = m * ev
    F['M9_amOdd'] = m * (1 - ev)
    F['xsE'] = x[:, 0::2].sum(1)
    for j in range(10):
        F['x%d' % j] = x[:, j]
        F['M%d' % j] = M[:, j]
        F['T%d' % j] = T[:, j]
        F['Mam%d' % j] = m * (am == j)
    srt = -np.sort(-x, 1)
    F['g2'] = srt[:, 0] - srt[:, 1]              # margin between the max and the second-largest
    S8 = -np.sort(-x[:, :9], 1)[:, 1]            # second-largest of x0..x8
    F['Q9'] = np.maximum(S8 - x[:, 9], 0)        # amount by which x9 falls below that
    def Tset(S): return T[:, list(S)].sum(1)
    def amset(S): return np.isin(am, list(S)) * 1.0
    return F, Tset, amset

def _terms(i, x):
    F, Tset, amset = _feats(x)
    if i == 0:  return [Tset([0,4,5,7]), Tset([1,6]), F['g2']*amset([1,3,5,7,8,9]), F['g2']*amset([9]), F['M9']*amset([0,2,3,5,7,9])]
    if i == 1:  return [F['M8'], F['M7'], F['negsum'], F['g2']*amset([0,2,4,6]), F['Q9']]
    if i == 2:  return [Tset([0,7,8]), Tset([0,1,3,4,5,6,7,8,9]), F['x8'], F['g2']*amset([5,7,8,9]), F['g2']*amset([9])]
    if i == 3:  return [F['T9'], F['g2']*amset([0,1,3,5,7]), F['g2']*amset([0,1,4,6,8])]
    if i == 4:  return [Tset(range(1,10)), Tset([0,2,3,4,6,9]), F['x5'], F['g2']*amset([0,1,3,7,9]), F['g2']*amset([8,9])]
    if i == 5:  return [F['G9'], F['Mam9']]
    if i == 6:  return [F['T9'], F['g2']*amset([1,3,5,7,8]), F['g2']*amset([2,4,6,8])]
    if i == 7:  return [Tset([1,3,4,5,6,7,8]), Tset([1,8]), F['g2']*amset([8,9]), F['g2']*amset([0,1,2,3,5,6,7,9]), F['Q9']]
    if i == 8:  return [F['T7'], F['g2']*amset([0,1,3,5,9]), F['g2']*amset([0,1,4,6])]
    if i == 9:  return [Tset([0,1,2,3,4,5,7,8,9]), Tset([1,3,5,6,8,9]), F['g2']*amset([7,8,9]), F['g2']*amset([0,1,3,5,6,9]), F['M9']*amset([0,2,5,7,8,9])]
    if i == 10: return [Tset([3,5,7,8]), Tset([5,8]), F['g2']*amset([9]), F['g2']*amset([1,5,8,9]), F['negsum']]
    if i == 11: return [Tset([2,3,4,6,9]), F['g2']*amset([0,1,2,4,6,8,9]), F['g2']*amset([0,1,2,3,4,5,7]), F['g2']*amset([4,9]), F['negsum']]
    if i == 12: return [F['T9'], F['g2']*amset([0,2,4,7]), F['g2']*amset([1,3,4])]
    if i == 13: return [Tset([0,2,5,6,7,8]), F['T4'], F['x2'], F['g2']*amset([0,7,8,9]), F['g2']*amset([0,2,3,4,5,7,9])]
    if i == 14: return [F['T4'], F['x9'], F['g2']*amset([0,2,7,8,9]), F['M9']*amset([1,3,4,5,6,7,9]), F['M9']*amset([1,3])]
    if i == 15: return [Tset([4,5,8]), Tset([1,4,5,6,7,8]), F['T8'], F['g2']*amset([0,2,6,8,9]), F['g2']*amset([9])]

COEFS = {0: [7.641, 2.826, -14.29, -16.58, -5.687], 1: [12.2, 20.1, 2.399, -7.389, -1.771], 2: [5.84, 2.636, -4.548, -15.3, -30.17], 3: [9.966, -12.04, 7.774], 4: [7.586, -2.457, -6.916, -16.65, -33.69], 5: [7.896, 0.9207], 6: [12.84, -7.251, 2.736], 7: [6.347, 5.699, -35.18, -15.22, -4.206], 8: [12.38, -13.2, 7.332], 9: [8.588, -5.0, -23.36, -12.99, -5.072], 10: [10.6, -6.021, -16.29, -13.26, 1.751], 11: [4.366, -14.86, 10.62, -8.552, 1.16], 12: [11.79, -8.991, 3.703], 13: [7.248, 5.192, -5.699, -23.04, -9.008], 14: [7.003, -2.136, -7.437, 2.119, 2.967], 15: [5.867, 2.821, 5.861, -14.25, -19.03]}

def f(i, x):
    t = _terms(i, x)
    return sum(c * v for c, v in zip(COEFS[i], t))

# "max" = largest of x0..x9; "gap of xj" = max - xj; "margin" = max minus second-largest of x0..x9.
LABELS = {
 0: ("B", "Grows with how far x0, x4, x5, x7 (weaker x1, x6) fall below the max of x0..x9; reduced by the margin (max minus second-largest) when the max is at x1, x3, x5, x7, x8, strongly at x9."),
 1: ("A", "Memory of the running maximum: about 12*max(x0..x8) + 20*max(x0..x7), plus a small boost from the total size of negative inputs; reduced by the margin (max minus second-largest) when the max is at x0, x2, x4 or x6."),
 2: ("B", "Grows with how far x0, x7 and especially x8 fall below the max of x0..x9 (other positions weakly); reduced by the margin (max minus second-largest) when the max is at x5, x7, x8, and strongly when it is x9."),
 3: ("A", "How far x9 falls below the max of x0..x9 (zero if x9 is the max), corrected by the margin (max minus second-largest): lowered when the max is at x3, x5, x7, raised at x4, x6, x8."),
 4: ("B", "Grows with how far x1, x5, x7, x8 (less other positions except x0) fall below the max of x0..x9; reduced by the margin (max minus second-largest) when the max is at x0, x1, x3, x7, strongly at x8, x9."),
 5: ("A", "Fires only when x9 is a new maximum: proportional to how far x9 exceeds the largest of x0..x8 (plus a small term proportional to x9 itself); zero otherwise."),
 6: ("A", "How far x9 falls below the max of x0..x9 (zero if x9 is the max), lowered by part of the margin (max minus second-largest) when the max sits at x1, x3, x5, x7 or x8."),
 7: ("B", "Grows with how far x1 and x8 (weaker x3..x7) fall below the max of x0..x9; reduced by the margin (max minus second-largest), strongly when the max is x8 or x9, and when x9 is low."),
 8: ("A", "How far x7 falls below the max of x0..x9 (zero if x7 is the max), corrected by the margin (max minus second-largest): lowered when the max is at x3, x5, x9, raised at x4, x6."),
 9: ("B", "Grows with how far x0, x2, x4, x7 (weaker other positions) fall below the max of x0..x9, but larger when x6 is near the max; reduced by the margin (max minus second-largest) when the max is late."),
 10: ("B", "Grows with how far x3 and x7 (weaker x5, x8) fall below the max of x0..x9; reduced by the margin (max minus second-largest) when the max is at x1, x5, x8, x9; small boost from negative inputs."),
 11: ("B", "Grows with how far x2, x3, x4, x6, x9 fall below the max of x0..x9; plus the margin (max minus second-largest) when the max is at x3, x5, x7, minus it when at x4, x6, x8, x9."),
 12: ("A", "How far x9 falls below the max of x0..x9 (zero if x9 is the max), lowered by part of the margin (max minus second-largest) when the max sits at x0, x2, x4 or x7 (slightly raised at x1, x3)."),
 13: ("B", "Grows with how far x0, x2 (strongest), x4..x8 fall below the max of x0..x9; reduced by the margin (max minus second-largest), strongly when the max is at x0, x7, x8, x9, mildly at x2..x5."),
 14: ("B", "Mainly how far x4 falls below the max of x0..x9 (zero if x4 is the max); plus a bonus proportional to the max when it is at x1 or x3; minus the max-minus-second-largest margin when the max is late."),
 15: ("B", "Grows with how far x8 and x4, x5 (weaker x1, x6, x7) fall below the max of x0..x9; reduced by the margin (max minus second-largest) when the max is at x0, x2, x6, x8, strongly at x9."),
}

CONSTANTS = {i: len(COEFS[i]) for i in range(16)}

_S = ("Linear, no intercept (the network is positively homogeneous). Position sets / which features were chosen by "
      "greedy+swap subset search over interpretable features (gaps max-xj, set sums of gaps, margin*[argmax in set], "
      "max*[argmax in set], running maxima, x9-vs-max(x0..x8) ReLUs); sets came from sorting coefficients of a full "
      "per-position linear fit and cutting at contiguous points. ")
STRUCT = {
 0: _S + "gap sets {0,4,5,7},{1,6}; margin sets {1,3,5,7,8,9},{9}; max set {0,2,3,5,7,9}",
 1: _S + "features max(x0..x8), max(x0..x7), sum of negative parts, margin*[argmax in {0,2,4,6}], ReLU(second-largest(x0..x8)-x9)",
 2: _S + "gap sets {0,7,8},{0,1,3,4,5,6,7,8,9}; x8; margin sets {5,7,8,9},{9}",
 3: _S + "gap of x9; margin sets {0,1,3,5,7},{0,1,4,6,8}",
 4: _S + "gap sets {1..9},{0,2,3,4,6,9}; x5; margin sets {0,1,3,7,9},{8,9}",
 5: _S + "ReLU(x9-max(x0..x8)) and max*[argmax=9]",
 6: _S + "gap of x9; margin sets {1,3,5,7,8},{2,4,6,8}",
 7: _S + "gap sets {1,3,4,5,6,7,8},{1,8}; margin sets {8,9},{0,1,2,3,5,6,7,9}; ReLU(second-largest(x0..x8)-x9)",
 8: _S + "gap of x7; margin sets {0,1,3,5,9},{0,1,4,6}",
 9: _S + "gap sets {0,1,2,3,4,5,7,8,9},{1,3,5,6,8,9}; margin sets {7,8,9},{0,1,3,5,6,9}; max set {0,2,5,7,8,9}",
 10: _S + "gap sets {3,5,7,8},{5,8}; margin sets {9},{1,5,8,9}; sum of negative parts",
 11: _S + "gap set {2,3,4,6,9}; margin sets {0,1,2,4,6,8,9},{0,1,2,3,4,5,7},{4,9}; sum of negative parts",
 12: _S + "gap of x9; margin sets {0,2,4,7},{1,3,4}",
 13: _S + "gap set {0,2,5,6,7,8}; gap of x4; x2; margin sets {0,7,8,9},{0,2,3,4,5,7,9}",
 14: _S + "gap of x4; x9; margin set {0,2,7,8,9}; max sets {1,3,4,5,6,7,9},{1,3}",
 15: _S + "gap sets {4,5,8},{1,4,5,6,7,8}; gap of x8; margin sets {0,2,6,8,9},{9}",
}
