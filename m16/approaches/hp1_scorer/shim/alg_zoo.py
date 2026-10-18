"""Loader shim (T281): this container has no alg_zoo install. Some submissions import m16/common.py, which loads
M_16,10 at import time via alg_zoo.example_2nd_argmax(). This shim returns the same weights from the HP1 panel's
'orig' entry (make_hp1.py copied them from common.py, so they are bit-identical in float64). Submissions that
rebind the weights per model (e.g. T286 probe._bind) never use them for scoring. Use: PYTHONPATH=<this dir>."""
import os
import numpy as np
import torch as th

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "data", "hp1", "weights.npz")


class _NS:
    pass


def example_2nd_argmax():
    z = np.load(_D)
    k = list(z["names"]).index("orig")
    m = _NS(); m.rnn = _NS(); m.linear = _NS()
    m.rnn.weight_ih_l0 = th.tensor(z["Wi"][k][:, None])
    m.rnn.weight_hh_l0 = th.tensor(z["Wh"][k])
    m.linear.weight = th.tensor(z["Wo"][k])
    m.eval = lambda: m
    return m
