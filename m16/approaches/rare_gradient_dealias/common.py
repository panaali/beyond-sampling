"""Shared helpers for T386: models from T372's frozen dirs files (weights w, directions, t0), truths, scoring."""
import json
import os

import numpy as np

import t372_rnn as rnn

HERE = os.path.dirname(os.path.abspath(__file__))
SH = [(16,), (16, 16), (10, 16)]
R = lambda f: json.load(open(os.path.join(HERE, "results", f)))


def case(nm):
    D = R(f"dirs_{nm}.json"); w = np.array(D["w"])
    return rnn.unflat(w, SH), D["t0"], np.array(D["dirs"]), D


def judge(pred, fd, se):
    """T372's go-line rule: resolved if |FD| > 3 s.e.; pass if same sign and .5 <= pred / FD <= 2."""
    res = abs(fd) > 3 * se; r = pred / fd
    return bool(res), bool(res and np.sign(pred) == np.sign(fd) and .5 <= r <= 2), float(r)
