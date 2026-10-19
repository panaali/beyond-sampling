import numpy as np

# ---- interpretable features of x (N,10) ----
def _top(x, k):
    """k-th largest (k=1,2) of each row."""
    return np.sort(x, 1)[:, -k]

def _DR(x, t):
    """drop below running max: relu(max(x0..x_{t-1}) - x_t)."""
    return np.maximum(x[:, :t].max(1) - x[:, t], 0)

def _DA(x, t):
    """drop below the mean of the two largest earlier values: relu(mean(top2 of x0..x_{t-1}) - x_t)."""
    s = np.sort(x[:, :t], 1)
    ref = 0.5 * (s[:, -1] + s[:, -2]) if t >= 2 else s[:, -1]
    return np.maximum(ref - x[:, t], 0)

def _design(i, x):
    x = np.asarray(x, float); N = len(x); one = np.ones(N)
    M, S = _top(x, 1), _top(x, 2)
    P = np.maximum(x[:, :3], 0).sum(1)  # positive mass of x0..x2
    if i == 0:
        return np.c_[_DR(x, 4) + _DR(x, 5), _DR(x, 3), x[:, 8], M + S, one]
    if i == 1:
        return np.c_[_DA(x, 6), _DA(x, 4), x[:, :9].max(1), P, one]
    if i == 2:
        hi = _DA(x, 3) + _DA(x, 5)
        mid = _DR(x, 1) + _DA(x, 2) + _DA(x, 4) + _DA(x, 6) + _DA(x, 9)
        lo = _DA(x, 7) + _DA(x, 8)
        return np.c_[hi, mid, lo, x[:, :9].max(1), one]
    if i == 3:
        return np.c_[np.abs(x[:, :3]).sum(1), _DR(x, 3), _DR(x, 5), _DA(x, 7), one]
    if i == 4:
        return np.c_[_DA(x, 8), x[:, 9], np.maximum(x, 0).sum(1), one]
    if i == 5:
        s9 = np.sort(x[:, :9], 1)
        return np.c_[s9[:, -1] + s9[:, -2], x[:, 9], np.maximum(x, 0).sum(1), one]
    if i == 6:
        return np.c_[M + S, _DR(x, 4), _DR(x, 2), x[:, 8], one]
    if i == 7:
        return np.c_[P, _DA(x, 5), _DR(x, 6), x[:, 9], one]

C = {0: [1.9576, 1.1416, 1.2161, 0.6631, 0.8571], 1: [1.9351, 0.6924, 0.7917, 0.5849, 0.0473], 2: [0.9996, 0.3369, -0.1198, 1.4703, 0.0634], 3: [0.4802, 1.0057, 1.7604, -2.7533, 0.2604], 4: [1.3763, -0.2933, 0.087, -0.1431], 5: [0.2411, -1.2654, 0.1932, -0.0425], 6: [0.7415, 1.9484, 0.6716, 0.3624, 0.464], 7: [1.5587, 2.7692, 1.5431, 1.3617, 1.604]}

def f(i, x):
    return np.maximum(_design(i, x) @ np.asarray(C[i]), 0.0)

LABELS = {
    0: ("A", "Rises when x3, x4, x5 each fall below the largest value before them (big early maximum), "
             "plus x8 and the sum of the two largest inputs; ReLU of that weighted sum."),
    1: ("A", "Mainly how far x6 falls below the mean of the two largest earlier values; smaller such term "
             "for x4; plus max of x0..x8 and positive mass of x0..x2. ReLU."),
    2: ("A", "Size of the largest of x0..x8 plus how far later inputs (mostly x3 and x5) fall below the "
             "mean of the two largest values preceding each. ReLU."),
    3: ("B", "Grows when x3 and x5 fall below the earlier running maximum; shrinks when x7 falls below the "
             "mean of the two largest earlier values (rewards a large x7). Plus |x0|+|x1|+|x2|. ReLU."),
    4: ("A", "How far x8 falls below the mean of the two largest of x0..x7 (x8 is not a top candidate), "
             "slightly lowered by x9 and raised by the total positive mass of all inputs. ReLU."),
    5: ("A", "How far x9 falls below the level of the two largest of x0..x8: ReLU(0.24*(top1+top2) "
             "- 1.27*x9 + 0.19*sum of positive inputs). Large means x9 is not a top-two value."),
    6: ("A", "Mainly how far x4 falls below the largest value before it, plus a weaker such term for x2, "
             "plus the sum of the two largest inputs and x8. ReLU."),
    7: ("A", "Mainly how far x5 falls below the mean of the two largest earlier values, plus how far x6 "
             "falls below the earlier maximum, plus x9 and positive mass of x0..x2. ReLU."),
}

CONSTANTS = {i: len(C[i]) for i in range(8)}

STRUCT = {
    0: "positions {3},{4,5} (drop-below-running-max), x8, magnitude feature (top1+top2) chosen by greedy/beam search",
    1: "positions 6,4 (drop below mean of earlier top two); magnitude = max(x0..x8) and relu-sum of x0..x2 (window x0..x2 searched)",
    2: "3 weight classes of drop positions: {3,5}, {1,2,4,6,9}, {7,8} (quantized from a full per-position fit); magnitude = max(x0..x8)",
    3: "drop positions 3,5 (running max) and 7 (mean of top two); |x0|+|x1|+|x2| (window searched)",
    4: "drop position 8 (mean of top two of x0..x7), x9, relu-sum of all inputs",
    5: "top two of x0..x8, x9, relu-sum of all inputs",
    6: "drop positions 4,2 (running max), x8, magnitude = top1+top2",
    7: "drop positions 5 (mean top two), 6 (running max), x9, relu-sum of x0..x2 (window searched)",
}
