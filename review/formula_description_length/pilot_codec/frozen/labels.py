"""Neuron labels for the 2nd-argmax RNN (16 ReLU units, 10 steps).

Notation used throughout (all computed from x only):
  x9          the last input x[:,9] (the only input the "clock" units have not yet absorbed)
  m1, p1      value and position of the largest of the first nine inputs x[:,0:9]
  m2, p2      value and position of the second largest of x[:,0:9]
  gap         m1 - m2
  x8, R7      the second-to-last input and the running max of x[:,0:8]
  S(p)        |sin(w (p - phi))|   circular distance of a position from a preferred position phi
  T(p)        sin^2(w (p - phi))   same idea, raised-cosine shape
  C(p)        cos^2(w (p - phi))   bump centred on phi
The network is positively homogeneous (ReLU, no biases), so every formula is degree-1 in x.
Each neuron uses at most 5 fitted constants (listed in its function).
"""
import numpy as np

LABELS = {
    0: ("A", "Position clock for the earlier max (preferred/zero position ~7, |sin| period ~8): "
             "~ 22.7*|sin(.41(p1-6.86))|*(m1-m2) + m2*(0.9*|sin(.41(p2-6.86))| + 5.0*|sin(.41(p1-6.86))|). "
             "Silent when the max of x0..x8 is at position 7, largest when it is at ~3; scaled by the top-2 gap, "
             "plus a smaller term from where the 2nd value sits."),
    1: ("B", "Last-input comparator: fires by ~10.4*(m2 - x9), i.e. while x9 is below the running 2nd-largest "
             "(so x9 is NOT the 2nd max), plus a position-modulated multiple of the top-2 gap "
             "(gain peaks when the max is at position ~3, ~0 at position ~7)."),
    2: ("A", "Leaky running maximum of x0..x8: ~18*m2 + (14.6 + 4.4cos(.65(p1-6.3)))*(m1-m2); i.e. ~15-18x the max, "
             "with older maxima slightly decayed. Always on; ignores x9."),
    3: ("C", "Fast/irregular position code voting for answer positions {0,1,6}: ~ m2*(1.7 + 4.9*[#of p1,p2 in {0,1,6}]) "
             "+ small gap term. Only partly captured; also depends on the 3rd value and other details."),
    4: ("A", "New-record detector at the last step: ~10.2*(x9 - m2) - (8.5 - 1.8cos(.68(p1-2.4)))*(m1-m2), "
             "i.e. ~10*relu(x9 - (m2 + ~0.85*(m1-m2))): positive only when x9 beats (about) the earlier max."),
    5: ("A", "x8 was a record/near-record: ~13.6*x8 minus a position-weighted sum of the positive parts of x0..x7 "
             "(weights 0.6-3.7, heaviest at positions 4-5). Active (~29% of the time) mostly when x8 is the max or "
             "2nd max of x0..x8."),
    6: ("B", "Last-input comparator like n1: ~10.7*(m2 - x9) + (0.85 + 12.2|sin(.43(p1-8.05))|)*(m1-m2); "
             "gap gain peaks for max at position ~4-5, ~0 for max at position 8."),
    7: ("A", "Cleanest last-input comparator: ~13*(m2 - x9) + (10.8 + 3.2cos(.69(p1-7)))*(m1-m2), roughly 13*(m1 - x9) - 2*(m1-m2): "
             "how far x9 lies below the earlier maximum, with only a weak dependence on where the max was; zero when x9 is a new record or nearly one."),
    8: ("B", "Last-input comparator like n1/n6: ~12.3*(m2 - x9) + (7.8 + 9.1cos(.58(p1-1.1)))*(m1-m2); "
             "gap gain large for an early max (positions 0-3), ~0 for a late max (5-8)."),
    9: ("A", "x8 is (essentially) the max of x0..x8: ~1.6*x8 - 0.5*R7 when x8 > R7, plus a steep near-tie response "
             "~98.5*x8 - 97.4*R7 when x8 is just below R7. Small, rarely active (~11%)."),
    10: ("A", "Position clock, preferred/zero position ~2.6 (raised-cosine sin^2, period ~8.4): "
              "~23.8*T(p1)*((m1-m2) + 0.59*T(p2)*m2) + 1.0*m2. Silent when the max is at 2-3 (and reduced when the "
              "2nd max is there), large when the max is at 6-8 or 0."),
    11: ("A", "Position clock, preferred/zero position ~3.7: ~20.6*T(p1)*((m1-m2) + 0.55*T(p2)*m2) + 1.2*m2 "
              "(w=.42). Silent when max is at 4, large for max at 0 or 7-8."),
    12: ("A", "Position clock, preferred/zero position ~5.2: ~31*T(p1)*(m1-m2) + (7.9*(T(p1)+T(p2)) + 0.84)*m2 (w=.34). "
              "Silent when max (or 2nd) at 5, large for max at 0-2."),
    13: ("C", "Mixed position code voting for answer 5 and against 2/6: ~(4.6+0.38*p1)*(m1-m2) + "
              "m2*(5.3 + 3.3*[#of p1,p2 in {0,1,5}] - 1.7*[#of p1,p2 in {2,6}]). Partly understood."),
    14: ("A", "Position clock, preferred/zero position ~6.7 (|sin|, w=.45), nearly the same tuning as n0: "
              "~23.1*S(p1)*(m1-m2) + (5.1*(S(p1)+S(p2)) + 1.65)*m2; large when max at ~3."),
    15: ("B", "Slow clock + fast set code: ~18*cos^2(.33(p1-1.3))*(m1-m2) (bump for max at positions 0-3) + "
              "m2*(1.8 + 5.8*[#of p1,p2 in {0,1,2,5}])."),
}


def _feats(x):
    x = np.asarray(x, dtype=np.float64)
    f9 = x[:, :9]
    o = np.argsort(-f9, axis=1)
    r = np.arange(len(x))
    p1, p2 = o[:, 0], o[:, 1]
    return dict(m1=f9[r, p1], m2=f9[r, p2], p1=p1.astype(float), p2=p2.astype(float),
                x9=x[:, 9], x8=x[:, 8], R7=x[:, :8].max(1), RX=np.maximum(x, 0))


def _S(w, p, phi):
    return np.abs(np.sin(w * (p - phi)))


def _T(w, p, phi):
    return np.sin(w * (p - phi)) ** 2


def _C(w, p, phi):
    return np.cos(w * (p - phi)) ** 2


def _I(p, s):
    return np.isin(p, s).astype(float)


def _n0(F):
    A, w, ph, b, c = 22.6672, 0.4059, 6.8561, 0.9149, 5.0398
    s1, s2 = _S(w, F['p1'], ph), _S(w, F['p2'], ph)
    return A * s1 * (F['m1'] - F['m2']) + b * s2 * F['m2'] + c * s1 * F['m2']


def _comparator(F, a, b, c, w, ph, sinform):
    g = F['m1'] - F['m2']
    mod = np.abs(np.sin(w * (F['p1'] - ph))) if sinform else np.cos(w * (F['p1'] - ph))
    return a * (F['m2'] - F['x9']) + (b + c * mod) * g


def _n1(F):
    return _comparator(F, 10.3608, -2.5483, 17.3984, 0.4492, 6.6248, True)


def _n2(F):
    a, b, c, w, ph = 18.1231, 14.6038, 4.3792, 0.6479, 6.3285
    return a * F['m2'] + (b + c * np.cos(w * (F['p1'] - ph))) * (F['m1'] - F['m2'])


def _n3(F):
    a, b, c, d, e = 3.0577, -0.6598, 1.6764, 4.9438, -0.113
    S_ = [0, 1, 6]
    i1, i2 = _I(F['p1'], S_), _I(F['p2'], S_)
    return (a + b * i1) * (F['m1'] - F['m2']) + (c + d * (i1 + i2)) * F['m2'] + e * F['x9']


def _n4(F):
    return _comparator(F, -10.228, -8.4725, 1.8253, 0.6764, 2.43, False)


def _n5(F):
    a, b, c, w, ph = 13.5681, 2.171, 1.5528, 0.6807, 4.39
    wt = b + c * np.cos(w * (np.arange(8) - ph))
    return a * F['x8'] - (wt * F['RX'][:, :8]).sum(1)


def _n6(F):
    return _comparator(F, 10.7121, 0.8522, 12.1798, 0.4316, 8.0545, True)


def _n7(F):
    return _comparator(F, 13.0438, 10.8226, 3.2425, 0.6947, 7.0061, False)


def _n8(F):
    return _comparator(F, 12.3024, 7.8372, 9.1414, 0.5843, 1.1048, False)


def _n9(F):
    a, b, c = -96.9186, 98.5193, 97.3724
    return a * np.maximum(F['x8'] - F['R7'], 0) + b * F['x8'] - c * F['R7']


def _clock_T(F, A, w, ph, c, d):
    t1, t2 = _T(w, F['p1'], ph), _T(w, F['p2'], ph)
    return A * t1 * ((F['m1'] - F['m2']) + c * t2 * F['m2']) + d * F['m2']


def _n10(F):
    return _clock_T(F, 23.7723, 0.3761, 2.6189, 0.5875, 0.9956)


def _n11(F):
    return _clock_T(F, 20.626, 0.4191, 3.727, 0.5496, 1.199)


def _n12(F):
    A, w, ph, B, C = 31.0625, 0.3422, 5.2101, 7.9337, -0.8449
    t1, t2 = _T(w, F['p1'], ph), _T(w, F['p2'], ph)
    return A * t1 * (F['m1'] - F['m2']) + (B * (t1 + t2) - C) * F['m2']


def _n13(F):
    a, b, c, d, e = 4.5618, 0.3821, 5.2536, 3.3007, 1.6846
    up = _I(F['p1'], [0, 1, 5]) + _I(F['p2'], [0, 1, 5])
    dn = _I(F['p1'], [2, 6]) + _I(F['p2'], [2, 6])
    return (a + b * F['p1']) * (F['m1'] - F['m2']) + (c + d * up - e * dn) * F['m2']


def _n14(F):
    A, w, ph, B, C = 23.0936, 0.4463, 6.7093, 5.0648, -1.652
    s1, s2 = _S(w, F['p1'], ph), _S(w, F['p2'], ph)
    return A * s1 * (F['m1'] - F['m2']) + (B * (s1 + s2) - C) * F['m2']


def _n15(F):
    A, w, ph, b, c = 18.033, 0.3324, 1.3109, 1.8452, 5.8242
    S_ = [0, 1, 2, 5]
    return A * _C(w, F['p1'], ph) * (F['m1'] - F['m2']) + (b + c * (_I(F['p1'], S_) + _I(F['p2'], S_))) * F['m2']


_FUNCS = [_n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7, _n8, _n9, _n10, _n11, _n12, _n13, _n14, _n15]


def f(i, x):
    F = _feats(x)
    return np.maximum(_FUNCS[i](F), 0.0)
