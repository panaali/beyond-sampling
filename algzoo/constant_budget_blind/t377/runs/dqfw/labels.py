import numpy as np

# ---------------------------------------------------------------- features
def _base(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    o = np.argsort(-x, 1, kind="stable")
    s = -np.sort(-x, 1)
    rank = np.argsort(o, 1)                     # rank[n, j] = 0 for the largest
    rm = np.maximum.accumulate(x, 1)            # rm[:, j] = max(x0..xj)
    s8 = -np.sort(-x[:, :9], 1)                 # sorted x0..x8
    s7 = -np.sort(-x[:, :8], 1)
    d = dict(x=x, N=N, am=o[:, 0], a2=o[:, 1], m1=s[:, 0], m2=s[:, 1], m3=s[:, 2], m4=s[:, 3],
             rank=rank, rm=rm, M8=rm[:, 8], M7=rm[:, 7], M5=rm[:, 5], M4=rm[:, 4], M2=rm[:, 2],
             S8=s8[:, 1], T8=s8[:, 2], S4=-np.sort(-x[:, :5], 1)[:, 1], T7=s7[:, 2],
             x9=x[:, 9])
    d["R8"] = np.maximum(x[:, 8] - rm[:, 7], 0)
    return d

relu = lambda z: np.maximum(z, 0)

def Q1(d, p):  # x_p if x_p is the largest, else 0
    return np.where(d["rank"][:, p] == 0, d["x"][:, p], 0.0)

def Q2(d, p):  # x_p if x_p is among the two largest
    return np.where(d["rank"][:, p] <= 1, d["x"][:, p], 0.0)

def Q3(d, p):
    return np.where(d["rank"][:, p] <= 2, d["x"][:, p], 0.0)

def _gate_x9neg(d, P, gate8):
    g = (d["am"] != 8) * 1.0 if gate8 else np.ones(d["N"])
    ip = np.isin(d["am"], P) * 1.0
    return [d["S8"] * g, d["M8"] * g, d["x9"] * g, d["S8"] * g * ip, d["M8"] * g * ip]

def _rankpos(d, p):
    rk = d["rank"][:, p]; xp = d["x"][:, p]
    so = d["x"].sum(1) - xp
    return [(rk == 0) * xp, (rk == 1) * xp, (rk == 2) * xp, (rk <= 2) * so]

def _notp(d, p, k, P, key):
    G = (d["rank"][:, p] >= k) * 1.0
    ip = np.isin(d[key], P) * 1.0
    return [G * d["m1"], G * d["m2"], G * d["x"][:, p], G * ip * d["m1"], G * ip * d["m2"]]

def _x9top2(d):
    return relu(d["x9"] - d["m3"])

def _other_top2_even(d):
    other = np.where(d["am"] == 9, d["a2"], d["am"])
    return (other % 2 == 0) * 1.0

def _other_top2_val(d, p):  # value of the other member of the top two, when x_p is one of them
    return np.where(d["am"] == p, d["m2"], d["m1"])

def _x9_21(d):
    t = (d["rank"][:, 9] <= 1) * 1.0
    ev = _other_top2_even(d)
    return [t * d["x9"], t * _other_top2_val(d, 9), t * d["m3"], t * ev * d["x9"], t * ev * d["m3"]]

def _notp_pos(d, p, k):
    G = (d["rank"][:, p] >= k) * 1.0
    return [G * d["m2"], G * d["x"][:, p], G * d["m2"] * d["a2"] / 9, G * d["m1"], G * d["m1"] * d["am"] / 9]

def _notp28(d):
    G = (d["rank"][:, 8] >= 1) * 1.0
    return [G * d["m1"], G * d["m1"] * d["am"] / 9, G * d["x"].sum(1), G * d["m2"] * d["am"] / 9,
            G * d["m1"] * (d["am"] % 2 == 0)]

# SPEC[i] = (mode, function d -> list of columns); prediction = sum c_k col_k, then ReLU if mode == 'relu'
SPEC = {
    0: ("relu", lambda d: _gate_x9neg(d, [0, 3, 4, 7], True)),
    1: ("lin", lambda d: [relu(d["M7"] - d["S8"])]),
    2: ("relu", lambda d: _rankpos(d, 3)),
    3: ("relu", lambda d: [d["M4"], Q2(d, 8), d["S4"], Q1(d, 3), Q1(d, 0)]),
    4: ("relu", lambda d: [(d["a2"] % 2 == 0) * (d["am"] != 9) * d["m2"], (d["a2"] % 2 == 0) * (d["am"] != 9) * d["m3"],
                           (d["am"] != 9) * d["m2"]]),
    5: ("relu", lambda d: [d["M8"] - d["S8"], d["R8"], d["M4"], d["x"][:, 2]]),
    6: ("relu", lambda d: _gate_x9neg(d, [0, 2, 5, 7], False)),
    7: ("lin", lambda d: [relu(d["x9"] - d["M8"])]),
    8: ("relu", lambda d: [np.abs(d["x"]).sum(1), relu(d["x"][:, 5] - d["m3"]), d["M2"], d["x"].sum(1), d["M5"]]),
    9: ("relu", lambda d: [d["S8"], d["T7"], d["x9"], d["m4"]]),
    10: ("relu", lambda d: [Q1(d, 4), Q1(d, 5), Q2(d, 8), Q2(d, 7), d["x9"]]),
    11: ("relu", lambda d: [Q1(d, 4), Q1(d, 0) + Q1(d, 6), Q1(d, 9)]),
    12: ("lin", lambda d: [relu(d["S8"] - d["x9"])]),
    13: ("relu", lambda d: _gate_x9neg(d, [0, 3, 5, 6], True)),
    14: ("relu", lambda d: [(d["rank"][:, 8] <= 1) * _other_top2_val(d, 8), (d["rank"][:, 8] <= 1) * d["m3"],
                            d["x"][:, 1:8].sum(1), relu(d["x"][:, 8] - d["m3"]), d["x"][:, 8]]),
    15: ("relu", lambda d: _rankpos(d, 1)),
    16: ("lin", lambda d: [_x9top2(d)]),
    17: ("lin", lambda d: [d["M8"], d["S8"]]),
    18: ("relu", lambda d: _gate_x9neg(d, [1], False)),
    19: ("relu", lambda d: _rankpos(d, 6)),
    20: ("relu", lambda d: _notp(d, 0, 2, [1, 8], "am")),
    21: ("relu", lambda d: _x9_21(d)),
    22: ("relu", lambda d: _rankpos(d, 7)),
    23: ("lin", lambda d: [_x9top2(d) * (d["R8"] == 0)]),
    24: ("relu", lambda d: _notp(d, 5, 2, [8, 9], "am")),
    25: ("relu", lambda d: _notp_pos(d, 8, 2)),
    26: ("relu", lambda d: [Q3(d, 2), d["M4"], relu(d["x"][:, 0] - d["m3"]), Q1(d, 2), Q1(d, 4)]),
    27: ("relu", lambda d: [Q1(d, 2) + Q1(d, 6), Q1(d, 9), relu(d["x"][:, 2] - d["m3"]), Q3(d, 7)]),
    28: ("relu", lambda d: _notp28(d)),
    29: ("relu", lambda d: _gate_x9neg(d, [0, 3, 6, 7], True)),
    30: ("relu", lambda d: _notp(d, 4, 2, [0, 3], "am")),
    31: ("relu", lambda d: [d["S8"], d["M7"], Q3(d, 2), d["M4"]]),
}

# BEGIN CONSTS
C = {
    0: [0.6712, 1.074, -1.528, 0.4212, -0.8136],
    1: [4.795],
    2: [3.013, 2.858, 1.946, -0.2525],
    3: [0.943, -2.867, -0.7564, 0.5926, 0.4781],
    4: [0.9685, -0.4324, 0.2339],
    5: [2.782, 3.039, 1.204, -0.6334],
    6: [1.304, 0.3459, -1.854, -0.8534, 1.191],
    7: [8.84],
    8: [0.1682, 2.105, 0.7677, -0.218, 0.9322],
    9: [-2.517, -0.7854, -0.2114, -0.03062],
    10: [0.5064, 0.4261, -1.013, -0.3651, -0.1328],
    11: [1.331, 0.73, 0.208],
    12: [3.119],
    13: [0.63, 1.372, -1.693, 0.5561, -1.017],
    14: [1.175, -1.007, -0.2779, -1.309, 1.38],
    15: [3.32, 2.509, 1.659, -0.2103],
    16: [5.708],
    17: [8.113, 2.041],
    18: [0.7948, 0.4151, -1.388, -1.047, 1.778],
    19: [2.296, 2.258, 1.205, -0.1688],
    20: [0.4586, 0.8073, -0.4748, 0.6076, -0.2197],
    21: [3.666, -2.083, -1.897, 0.8815, -0.2594],
    22: [2.435, 1.095, 0.4395, -0.1452],
    23: [4.252],
    24: [0.4512, 0.9288, -0.4817, -0.2775, -0.04067],
    25: [1.073, -0.2458, -0.6943, 0.3592, -0.2601],
    26: [1.967, 0.4871, 2.104, 1.492, 0.8926],
    27: [1.167, 0.2077, 0.2295, -0.2442],
    28: [1.463, -1.545, -0.08537, 1.159, -0.2096],
    29: [1.302, 0.1963, -1.633, -0.5132, 0.9912],
    30: [0.473, 1.195, -0.6527, 0.7637, -0.5498],
    31: [4.997, 0.666, -0.9092, 0.9455],
}
# END CONSTS

def f(i, x):
    d = _base(x)
    mode, fn = SPEC[i]
    cols = fn(d)
    c = C.get(i, [0.0] * len(cols))
    y = sum(ck * col for ck, col in zip(c, cols))
    y = np.asarray(y, dtype=float) * np.ones(d["N"])
    return relu(y) if mode == "relu" else y

LABELS = {0: ('A', 'Active when x9 is below the second-largest of x0..x8 and the maximum is not x8: a mix of the top two of x0..x8 minus 1.5-1.9 times x9, reweighted when the maximum is at x0, x3, x4, x7.'), 1: ('A', 'Gap between the largest and second-largest of x0..x8, zero when x8 itself is the largest of x0..x8: 4.8 x ReLU(max(x0..x7) - second-largest of x0..x8).'), 2: ('A', 'Fires when x3 is among the three largest inputs: a rank-dependent multiple of x3 (largest > second > third), minus about 0.15-0.25 times the sum of the other nine inputs.'), 3: ('D', 'Unclear. Loosely follows max(x0..x4) minus the second-largest of x0..x4, larger when the second-largest overall sits at x5-x7 and the max earlier; silent when x8 is in the top two.'), 4: ('D', 'Weak parity signal: larger when the second-largest input is at an even position (x0,x2,x4,x6,x8) and x9 is not the maximum; scales with the second-largest value. Poorly fit.'), 5: ('A', 'Always on: gap between largest and second-largest of x0..x8, plus extra when x8 sets a new running maximum (x8 minus max of x0..x7), plus a max(x0..x4) term.'), 6: ('A', 'Active when x9 is below the second-largest of x0..x8: a mix of the top two of x0..x8 minus 1.5-1.9 times x9, reweighted when the maximum is at x0, x2, x5, x7.'), 7: ('A', 'How much x9 exceeds every earlier input: 8.8 x ReLU(x9 - max(x0..x8)); zero unless x9 is the largest.'), 8: ('C', 'Always on: generic magnitude (sum of |x|, minus sum of x, running maxima of x0..x2 and x0..x5) plus a strong boost when x5 is one of the two largest inputs.'), 9: ('A', 'Fires only when inputs are mostly negative: ReLU of -(2.5 x second-largest of x0..x8 + 0.8 x third-largest of x0..x7 + 0.2 x x9).'), 10: ('A', 'Small signal that the maximum is at x4 or x5 (about 0.5 x the max), suppressed when x7 or x8 is among the two largest.'), 11: ('A', "Codes the maximum's position: about 1.3 x max if the max is x4, 0.7 x max if it is x0 or x6, 0.2 x max if it is x9, otherwise zero."), 12: ('A', 'How far x9 falls below the second-largest of x0..x8: 3.1 x ReLU(second-largest(x0..x8) - x9); zero when x9 is one of the two largest.'), 13: ('A', 'Active when x9 is below the second-largest of x0..x8 and the maximum is not x8: a mix of the top two of x0..x8 minus 1.5-1.9 times x9, reweighted when the maximum is at x0, x3, x5, x6.'), 14: ('A', 'Tracks x8 being one of the two largest: then about 1.2 x the other top-two value + 0.3 x third-largest - 0.28 x sum(x1..x7); otherwise weak ReLU(1.4 x8 - 0.28 sum(x1..x7)).'), 15: ('A', 'Fires when x1 is among the three largest inputs: a rank-dependent multiple of x1 (largest > second > third), minus about 0.15-0.25 times the sum of the other nine inputs.'), 16: ('A', '5.7 x ReLU(x9 - third-largest input): positive exactly when x9 is one of the two largest, measuring its lead over the third.'), 17: ('A', 'Magnitude of the top of x0..x8: about 8.1 x max(x0..x8) + 2.0 x second-largest(x0..x8); x9 is ignored.'), 18: ('A', 'Active when x9 is below the second-largest of x0..x8: a mix of the top two of x0..x8 minus 1.5-1.9 times x9, reweighted when the maximum is at x1.'), 19: ('A', 'Fires when x6 is among the three largest inputs: a rank-dependent multiple of x6 (largest > second > third), minus about 0.15-0.25 times the sum of the other nine inputs.'), 20: ('A', 'Silent when x0 is one of the two largest; otherwise roughly 0.5 x max + 0.8-1.2 x second-largest - 0.25-0.5 x x0, reweighted when the max is at x1 or x8.'), 21: ('B', 'Active when x9 is one of the two largest: about 3.7 x x9 - 2.1 x (other top-two value) - 1.9 x third-largest, boosted when the other top-two input sits at an even position.'), 22: ('A', 'Fires when x7 is among the three largest inputs: a rank-dependent multiple of x7 (largest > second > third), minus about 0.15-0.25 times the sum of the other nine inputs. Strongest when x7 is the maximum.'), 23: ('A', "4.25 x ReLU(x9 - third-largest input), i.e. x9's lead when it is one of the two largest, but zero when x8 exceeds all of x0..x7."), 24: ('A', 'Silent when x5 is one of the two largest; otherwise roughly 0.5 x max + 0.8-1.2 x second-largest - 0.25-0.5 x x5, reweighted when the max is at x8 or x9.'), 25: ('A', 'Silent when x8 is one of the two largest; otherwise roughly 0.5 x max + 0.8-1.2 x second-largest - 0.25-0.5 x x8, shrinking the later the max and second-largest occur.'), 26: ('B', 'Mainly signals x2 being among the three largest (about 2 x x2, more if x2 is the max); also responds to x0 in the top two and the max at x4, plus a max(x0..x4) term.'), 27: ('B', 'Signals the maximum being at x2 or x6 (about 1.17 x max), with small extras when the max is x9 or x2 is in the top three.'), 28: ('A', 'Silent when x8 is the maximum; otherwise about max x (1.46 - 0.17 x position of max) + 0.13 x position x second-largest, i.e. larger when the max came early.'), 29: ('A', 'Active when x9 is below the second-largest of x0..x8 and the maximum is not x8: a mix of the top two of x0..x8 minus 1.5-1.9 times x9, reweighted when the maximum is at x0, x3, x6, x7.'), 30: ('A', 'Silent when x4 is one of the two largest; otherwise roughly 0.5 x max + 0.8-1.2 x second-largest - 0.25-0.5 x x4, reweighted when the max is at x0 or x3.'), 31: ('A', 'Second-largest of x0..x8 (about 5x), plus smaller max(x0..x7) and max(x0..x4) terms, reduced when x2 is among the three largest.')}

STRUCT = {0: 'argmax position set {0,3,4,7} found by greedy search; gate argmax!=8', 13: 'argmax set {0,3,5,6} (greedy); gate argmax!=8', 29: 'argmax set {0,3,6,7} (greedy); gate argmax!=8', 6: 'argmax set {0,2,5,7} (greedy search)', 18: 'argmax set {1} (greedy search)', 2: 'position 3; rank<=2 gating', 15: 'position 1; rank<=2 gating', 19: 'position 6; rank<=2 gating', 22: 'position 7; rank<=2 gating', 20: 'p=0, rank>=2 gate, argmax set {1,8} (greedy)', 24: 'p=5, rank>=2 gate, argmax set {8,9} (greedy)', 30: 'p=4, rank>=2 gate, argmax set {0,3} (greedy)', 25: 'p=8, rank>=2 gate; features chosen by greedy search', 28: 'p=8, not-max gate; features chosen by greedy search', 3: '5 features chosen by greedy search', 8: '5 features chosen by greedy search', 9: '4 features chosen by greedy search', 10: 'positions {4},{5},{7},{8} chosen by greedy', 11: 'positions {4},{0,6},{9} chosen by greedy', 26: 'features chosen by greedy search', 27: 'positions {2,6},{9},{2},{7} chosen by greedy', 31: 'features chosen by greedy search', 5: 'features chosen by greedy search', 14: 'feature set chosen by inspection + search', 21: 'parity of the other top-two position (inspection)', 23: 'gate: x8 not a new running max', 4: 'parity of second-largest position; gate argmax!=9', 1: '', 7: '', 12: '', 16: '', 17: ''}

_x0 = np.random.default_rng(0).standard_normal((20, 10))
CONSTANTS = {i: len(SPEC[i][1](_base(_x0))) for i in range(32)}
