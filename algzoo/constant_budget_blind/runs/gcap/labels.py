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
SPEC = {0: (['LEVEL', 'x3', 'x4', 'x5', 'x8'], [0.7003, -0.9674, -1.8916, -2.0407, 1.2051]), 1: (['G6', 'LEVEL', 'D46', 'g6', 'D48'], [1.6465, 0.1552, -1.8684, 1.5168, 0.4385]), 2: (['LEVEL', 'x5', 'x3', 'g6', 'D89'], [0.366, -0.8258, -0.7175, 0.4857, 0.4865]), 3: (['G5', 'x7', 'D35', 'D47', 'D17'], [2.9973, 1.3379, -2.0243, -1.2359, -0.7614]), 4: (['gap8', 'gap9'], [1.49, 0.3077]), 5: (['gap9'], [1.2595]), 6: (['LEVEL', 'x2', 'x4', 'x8', 'D24'], [0.4625, -0.2763, -2.3356, 0.4098, -0.8412]), 7: (['LEVEL', 'x5', 'x6', 'x9', 'g4'], [0.6432, -2.4071, -1.4667, 1.2877, 0.8624])}

def f(i, x):
    names, coefs = SPEC[i]
    z = sum(c * feat(x, n) for n, c in zip(names, coefs))
    return np.maximum(z, 0.0)

LABELS = {
    0: ("B", "Magnitude of the running maximum (average of max(x0..xt) over t=1..9) minus how large x4 and x5 are (and x3, less), plus x8: large when x4 and x5 were low."),
    1: ("A", "How far x6 falls below the largest and second-largest of x0..x5, discounted when x4 is the larger reference value, plus a small running-maximum magnitude term."),
    2: ("C", "Running-maximum magnitude minus x5 and x3 (large when they were low), plus how far x6 falls below the second-largest of x0..x5, plus how much x8 exceeds x9."),
    3: ("B", "Mainly how far x5 falls below max(x0..x4), plus the value of x7; reduced when x3 exceeds x5, or when x4 or x1 exceed x7."),
    4: ("A", "How far x8 falls below the mean of the three largest of x0..x7, plus a small share of how far x9 falls below the mean of the three largest of x0..x8."),
    5: ("A", "How far x9 falls below the mean of the three largest of x0..x8 (zero when x9 is at least that high); signals that the last input is not a top value."),
    6: ("B", "Running-maximum magnitude (average of max(x0..xt), t=1..9) minus x4, more strongly when x4 is below x2; plus a little x8, minus a little x2. Large when x4 was low."),
    7: ("B", "Running-maximum magnitude minus x5 and x6 (large when they were low), plus x9, plus how far x4 falls below the second-largest of x0..x3."),
}

CONSTANTS = {i: len(SPEC[i][1]) for i in range(8)}

STRUCT = {
    0: "greedy/exhaustive search: level feature (avg running max) + 4 terms from {x_t, gap_t, G_t, g_t, ReLU(x_s-x_t)}; chose x3,x4,x5,x8",
    1: "greedy forward selection over ~130 features; chose G6, level, ReLU(x4-x6), g6, ReLU(x4-x8)",
    2: "greedy forward selection; chose level, x5, x3, g6, ReLU(x8-x9)",
    3: "greedy forward selection; chose G5, x7, ReLU(x3-x5), ReLU(x4-x7), ReLU(x1-x7)",
    4: "reference = mean of top-3 earlier values (compared top-1/2/3 and blends); positions 8,9",
    5: "reference = mean of top-3 of x0..x8 (compared top-1/2/3 and blends)",
    6: "exhaustive level + 3 terms, then best 5th; chose x2, x4, x8, ReLU(x2-x4)",
    7: "exhaustive level + 3 terms, then best 5th; chose x5, x6, x9, g4",
}
