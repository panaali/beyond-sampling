"""Separable position code: h_10 ~ sum of (scalar feature) x (vector indexed by one position)."""
import numpy as np
from state import discrete_state


def design(x, form):
    """Columns: one-hot(position) * scalar, for each (scalar, position-variable) term in form."""
    s = discrete_state(x)
    s["gap"] = s["m1"] - s["m2"]
    s["below9"] = s["m2"] - s["x9"]
    cols = []
    for scal, posvar in form:
        oh = np.eye(9)[s[posvar]] if posvar else np.ones((len(x), 1))
        cols.append(oh * s[scal][:, None])
    return np.concatenate(cols, 1), s


def fit(x, h, form, mask):
    X, _ = design(x, form)
    return np.linalg.lstsq(X[mask], h[mask], rcond=None)[0]
