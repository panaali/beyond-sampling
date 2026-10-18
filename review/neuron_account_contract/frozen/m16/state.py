"""Discrete 'state' of an input: positions of top-2 among x0..x8 and where x9 falls."""
import numpy as np


def discrete_state(x):
    head = x[:, :9]
    o = np.argsort(-head, 1)
    p1, p2, p3 = o[:, 0], o[:, 1], o[:, 2]
    r = np.arange(len(x))
    m1, m2, m3 = head[r, p1], head[r, p2], head[r, p3]
    x9 = x[:, 9]
    case = np.where(x9 > m1, 0, np.where(x9 > m2, 1, 2))  # 0: x9 is max, 1: x9 is 2nd, 2: below both
    return dict(p1=p1, p2=p2, p3=p3, m1=m1, m2=m2, m3=m3, x9=x9, case=case,
                key=(case * 9 + p1) * 9 + p2)
