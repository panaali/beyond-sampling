import numpy as np

# ---------- interpretable features of x (N,10), positions 0-based ----------
def _runmax(x):            # M[:,t] = max(x0..xt)
    return np.maximum.accumulate(x, axis=1)

def _topk_mean(cols, k):   # mean of the k largest values among the given columns
    s = -np.sort(-cols, axis=1)
    return s[:, :min(k, cols.shape[1])].mean(axis=1)

def feat(x, name):
    x = np.asarray(x, dtype=float)
    if name == 'LEVEL':    # average of the running maxima max(x0..xt), t=1..9, times 10
        return _runmax(x)[:, 1:].mean(axis=1) * 10.0
    if name.startswith('x'):
        return x[:, int(name[1:])]
    if name.startswith('gap'):   # how far x_t falls below the mean of the 3 largest of x0..x_{t-1}
        t = int(name[3:]); return np.maximum(_topk_mean(x[:, :t], 3) - x[:, t], 0)
    if name.startswith('G'):     # how far x_t falls below max(x0..x_{t-1})
        t = int(name[1:]); return np.maximum(x[:, :t].max(axis=1) - x[:, t], 0)
    if name.startswith('g'):     # how far x_t falls below the 2nd largest of x0..x_{t-1}
        t = int(name[1:]); s = -np.sort(-x[:, :t], axis=1); return np.maximum(s[:, 1] - x[:, t], 0)
    if name.startswith('D'):     # D_st = ReLU(x_s - x_t)
        s, t = int(name[1]), int(name[2]); return np.maximum(x[:, s] - x[:, t], 0)
    raise ValueError(name)

# SPEC[i] = (feature names, fitted coefficients); prediction = ReLU(sum coef*feature)
SPEC = __SPEC__

def f(i, x):
    names, coefs = SPEC[i]
    z = sum(c * feat(x, n) for n, c in zip(names, coefs))
    return np.maximum(z, 0.0)
