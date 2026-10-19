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

COEFS = __COEFS__

def f(i, x):
    t = _terms(i, x)
    return sum(c * v for c, v in zip(COEFS[i], t))
