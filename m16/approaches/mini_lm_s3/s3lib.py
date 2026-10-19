"""T446 (mini-LM S3) shared helpers. Imports the T433 harness read-only."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "..", "mini_lm")
sys.path.insert(0, HARNESS)
from langs import Dyck2, model_input  # noqa: E402
from load import load as _load, names  # noqa: E402
from metrics import logits_of  # noqa: E402

MODELS = os.environ.get("S3_MODELS")  # e.g. ../mini_lm/models_mlp1 for the MLP1 panel


def load(name):
    if MODELS or int(name.rsplit("_s", 1)[1]) >= 100:  # MLP1 seeds are 100-119
        return _load(name, models_dir=MODELS or os.path.join(HARNESS, "models_mlp1"))
    return _load(name)


DATA = os.path.join(HERE, "..", "..", "..", "data", "t446")
os.makedirs(DATA, exist_ok=True)
torch.set_num_threads(int(os.environ.get("S3_THREADS", "2")))

TAU_GRID = (0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
DYCK_MODELS = [f"dyck2_{a}_s{s}" for a in ("rnn", "softmax", "hard", "relu", "linear", "pos") for s in (0, 1, 2)]


def stack_info(y):
    """Per position t (predicting y[:, t]): depth d_t, top-of-stack type, and position of the matching opener."""
    N, L = y.shape
    depth = np.zeros((N, L), np.int64)
    top = np.full((N, L), -1, np.int64)
    opos = np.full((N, L), -1, np.int64)
    st = np.zeros((N, L + 1), np.int64)
    sp = np.zeros((N, L + 1), np.int64)
    d = np.zeros(N, np.int64)
    idx = np.arange(N)
    for t in range(L):
        depth[:, t] = d
        has = d > 0
        dm = np.maximum(d - 1, 0)
        top[has, t] = st[idx[has], dm[has]]
        opos[has, t] = sp[idx[has], dm[has]]
        tok = y[:, t]
        op = tok < 2
        st[idx[op], d[op]] = tok[op]
        sp[idx[op], d[op]] = t
        d = d + np.where(op, 1, -1)
    return depth, top, opos


def p_illegal(model, lang, y, chunk=8192):
    """(N, L) model probability mass on illegal next tokens at every position (float64)."""
    lg = torch.from_numpy(logits_of(model, lang, y, chunk=chunk))
    p = torch.softmax(lg, -1).numpy()
    depth, top, _ = stack_info(y)
    # illegal: both closers at depth 0; the wrong closer (3 - top) + 2 ... i.e. closer of the other type at d > 0
    wrong = np.where(top == 0, 3, 2)
    pw = np.take_along_axis(p, wrong[..., None], -1)[..., 0]
    return np.where(depth > 0, pw, p[..., 2] + p[..., 3])


def model_flops(model, L):
    return float(model.flops(L))


def node_cost(model, t, L):
    """FPE charge of computing position t (0-based) given a cached prefix: flops(t+1)-flops(t) over flops(L)."""
    return (model.flops(t + 1) - model.flops(t)) / model.flops(L)
