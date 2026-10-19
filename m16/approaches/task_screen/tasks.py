"""T419 task definitions: functions from x (N, 10) of N(0,1) inputs to a label in {0..9}.

Each task works on torch tensors (training, via alg_zoo's task_registry) and on numpy arrays (evaluation):
    import sys; sys.path.insert(0, "m16/approaches/task_screen")
    from tasks import TASKS, labels
    y = labels("argmax_absdiff", x)        # x: numpy (N, 10) float -> numpy int64 (N,)
Every label is a.s. unique: ties among continuous functions of Gaussian inputs have probability 0.
Positions use 0-based indices into the sequence. Tasks that look at (x_{i-1}, x_i) pairs use i = 1..9 (class 0 unused);
the 3-window uses i = 2..9.
"""
import numpy as np
import torch as th


def _kth_largest(k):
    return lambda x: x.argsort(-1)[..., -k]


def _pairs(x):
    return x[..., 1:], x[..., :-1]


def argmax_absdiff(x):            # largest |x_i - x_{i-1}|, i in 1..9
    a, b = _pairs(x)
    return (a - b).abs().argmax(-1) + 1


def argmin_absdiff(x):            # smallest |x_i - x_{i-1}| (closest adjacent pair), i in 1..9
    a, b = _pairs(x)
    return (a - b).abs().argmin(-1) + 1


def argmax_diff(x):               # largest signed jump x_i - x_{i-1}, i in 1..9
    a, b = _pairs(x)
    return (a - b).argmax(-1) + 1


def argmax_pairsum(x):            # largest x_i + x_{i-1}, i in 1..9
    a, b = _pairs(x)
    return (a + b).argmax(-1) + 1


def argmax_window3(x):            # largest x_{i-2} + x_{i-1} + x_i, i in 2..9
    return (x[..., 2:] + x[..., 1:-1] + x[..., :-2]).argmax(-1) + 2


def second_argmax_abs(x):         # 2nd largest |x_i|
    return x.abs().argsort(-1)[..., -2]


def argmax_abs(x):                # largest |x_i|
    return x.abs().argmax(-1)


def argmin_abs(x):                # value closest to 0
    return x.abs().argmin(-1)


def argmax_cumsum(x):             # where the running sum S_i = x_0 + ... + x_i peaks
    return x.cumsum(-1).argmax(-1)


def argoutlier(x):                # alg_zoo's argoutlier: largest |x_i - mean(x)|
    return (x - x.mean(-1, keepdim=True)).abs().argmax(-1)


def n_records(x):                 # number of left-to-right maxima (records) minus 1, in 0..9
    cm = x.cummax(-1).values if isinstance(x, th.Tensor) else np.maximum.accumulate(x, -1)
    return (x[..., 1:] > cm[..., :-1]).sum(-1)


TASKS = {
    "argmax": _kth_largest(1),
    "2nd_argmax": _kth_largest(2),      # M_16,10's task (control)
    "3rd_argmax": _kth_largest(3),
    "4th_argmax": _kth_largest(4),
    "5th_argmax": _kth_largest(5),      # = alg_zoo's argmedian at T = 10 (argsort index 5)
    "2nd_argmin": lambda x: x.argsort(-1)[..., 1],   # = 2nd_argmax of -x (symmetry control)
    "argmax_absdiff": argmax_absdiff,
    "argmin_absdiff": argmin_absdiff,
    "argmax_diff": argmax_diff,
    "argmax_pairsum": argmax_pairsum,
    "argmax_window3": argmax_window3,
    "2nd_argmax_abs": second_argmax_abs,
    "argmax_abs": argmax_abs,
    "argmin_abs": argmin_abs,
    "argmax_cumsum": argmax_cumsum,
    "argoutlier": argoutlier,
    "n_records": n_records,
}


def labels(name, x):
    """numpy (N, 10) -> numpy int64 labels (computed in torch float64 so both paths agree exactly)."""
    return TASKS[name](th.from_numpy(np.asarray(x, dtype=np.float64))).numpy().astype(np.int64)


def register():
    """Add every task to alg_zoo's task_registry (continuous inputs, 10-way 'dist' output), as alg_zoo's own tasks."""
    from alg_zoo.tasks import task_registry
    for k, f in TASKS.items():
        task_registry.setdefault("t419_" + k, (f, "continuous", "dist"))
