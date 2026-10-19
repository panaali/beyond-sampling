"""Per-neuron labels and formulas for h_10 of the second-largest-position RNN (numpy only)."""
import numpy as np
def relu(z): return np.maximum(z,0)
def topk(x,L,k):
    """k-th largest (0=largest) among x0..x_{L-1}"""
    return -np.sort(-x[:,:L],1)[:,k]
def oth9(x,j):
    o=np.delete(x[:,:9],j,1) if j<9 else x[:,:9]
    return -np.sort(-o,1)
def Mo9(x,j): return oth9(x,j)[:,0]
def So9(x,j): return oth9(x,j)[:,1]
def am9(x): return np.argmax(x[:,:9],1)
def a2_9(x): return np.argsort(-x[:,:9],1)[:,1]
def I1(x,j): return (am9(x)==j)*1.0
def I2(x,j): return (a2_9(x)==j)*1.0
def top2(x,j): return I1(x,j)+I2(x,j)
def J1(x,j): return (np.argmax(x,1)==j)*1.0
def so(x,j): return x[:,:9].sum(1)-(x[:,j] if j<9 else 0)
def inset(v,S): return np.isin(v,S)*1.0
def rSo9(x,j): return relu(x[:,j]-So9(x,j))
def rMo9(x,j): return relu(x[:,j]-Mo9(x,j))
def amax(x): return np.argmax(x,1)
def asec(x): return np.argsort(-x,1)[:,1]
def s_all(x,k): return -np.sort(-x,1)[:,k]
def pair30(x):
    a=amax(x); b=asec(x); P=[3,5,6,7]
    return (((a==8)&np.isin(b,P))|((b==8)&np.isin(a,P)))*1.0


SPEC = {
    0: ('relu', ['x[:,2]', 'I1(x,2)', 'I2(x,2)', 'inset(am9(x),[0])'], None, [0.5589, 4.8256, 2.3012, 1.7795, 0.4872]),
    1: ('relu', ['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[0, 2, 3])'], None, [0.4286, 1.3616, -1.8707, 1.4139, -0.5035]),
    2: ('relu', ['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[8])'], None, [1.199, 0.8698, -1.9937, -5.6002, 0.2284]),
    3: ('gated', ['x[:,8]', 'topk(x,9,0)', 'topk(x,9,1)', 'inset(am9(x),[5,6])'], '1-top2(x,8)', [-1.4486, 0.1912, 1.3305, 0.7583, -0.0815]),
    4: ('relu', ['x[:,4]', 'I1(x,4)', 'I2(x,4)', 'inset(am9(x),[1,2])'], None, [0.4453, 3.2643, 2.3378, 1.0995, 0.4007]),
    5: ('relu', ['topk(x,9,0)', 'topk(x,9,1)', 'inset(am9(x),[1,5,7])'], None, [3.2961, 2.14, 1.007, -0.2061]),
    6: ('relu', ['topk(x,9,1)', 'topk(x,9,0)', 'x[:,9]'], None, [2.9462, 0.1895, -3.099, 0.0899]),
    7: ('relu', ['x[:,6]', 'so(x,6)', 'I1(x,6)', 'I2(x,6)'], None, [1.7702, -0.3589, 2.144, 1.8056, 0.2958]),
    8: ('relu', ['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[0, 2, 4, 6, 8])'], None, [0.7989, 1.0833, -1.8751, 1.2125, -0.5983]),
    9: ('scale', ['relu(x[:,8]-topk(x,8,0))'], None, [7.2529]),
    10: ('gated', ['x[:,8]', 'So9(x,8)', 'am9(x)', 'Mo9(x,8)'], '1-top2(x,8)', [0.002, 0.5892, -0.1149, 0.3654, 0.8021]),
    11: ('relu', ['rSo9(x,8)', 'I1(x,2)', 'rMo9(x,8)', 'topk(x,6,0)'], None, [2.0797, 0.9947, -1.7377, 0.2978, -0.2571]),
    12: ('relu', ['topk(x,9,0)', 'topk(x,9,1)', 'inset(am9(x),[0,2,3,4,6])'], None, [5.7465, 3.478, 0.6042, -0.1631]),
    13: ('relu', ['x[:,5]', 'I1(x,5)', 'I2(x,5)', 'rSo9(x,5)'], None, [0.4223, 1.9489, 2.2149, 1.2325, 0.7419]),
    14: ('gated', ['topk(x,9,0)', 'topk(x,9,1)', 'I1(x,3)'], 'inset(am9(x),[0,1,2,3,5])', [0.7802, -0.3751, 0.8222, 0.32]),
    15: ('relu', ['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[1, 4])'], None, [0.5309, 1.413, -2.0416, 1.6055, -0.431]),
    16: ('gated', ['x[:,4]', 'so(x,4)'], 'I1(x,4)', [0.9072, -0.1128, 0.1504]),
    17: ('relu', ['rSo9(x,8)', 'J1(x,6)', 'I2(x,7)', 'topk(x,5,0)'], None, [-5.5904, 1.0298, 0.7278, 0.3553, 0.0199]),
    18: ('relu', ['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[0, 1, 2, 3, 4, 5, 6])'], None, [-0.5777, -3.21, 3.9894, 5.1614, -5.015]),
    19: ('relu', ['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[2, 3, 5, 6, 8])'], None, [-1.5812, -2.8702, 4.5296, 0.7616, -0.5632]),
    20: ('relu', ['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[0, 1, 2, 3, 4, 5, 6, 7])'], None, [0.8478, 0.9924, -1.8031, 4.9896, -4.7573]),
    21: ('lin', ['x[:,:8].sum(1)', 'topk(x,9,0)', 'topk(x,9,1)', 'topk(x,9,2)'], None, [-0.6243, 1.3386, 1.434, 0.5191, 1.0329]),
    22: ('scale', ['relu(x[:,9]-topk(x,9,0))'], None, [6.4919]),
    23: ('gated', ['x[:,7]', 'so(x,7)'], 'I1(x,7)', [1.0662, -0.1368, -0.0687]),
    24: ('gated', ['x[:,7]', 'So9(x,7)', 'so(x,7)'], '1-top2(x,7)', [-1.1923, 2.1351, -0.0583, 0.1409]),
    25: ('relu', ['rSo9(x,1)', 'rSo9(x,3)', 'I1(x,1)', 'I2(x,1)'], None, [2.0072, 2.6214, 2.4674, 1.1823, 0.4862]),
    26: ('relu', ['topk(x,8,0)', 'topk(x,9,1)'], None, [6.7144, -6.5198, -0.0389]),
    27: ('relu', ['x[:,0]', 'so(x,0)', 'I1(x,0)', 'I2(x,0)'], None, [0.8189, -0.1193, 2.6166, 2.0969, 0.058]),
    28: ('gated', ['x[:,3]', 'topk(x,9,0)', 'topk(x,9,1)', 'inset(am9(x),[1,5,6,7,8])'], '1-top2(x,3)', [-0.5879, 0.5942, 0.6812, -0.6341, 1.0522]),
    29: ('relu', ['rSo9(x,1)', 'topk(x,4,0)', 'x[:,1]', 'rSo9(x,0)'], None, [-3.136, 1.5413, -0.7723, 0.8952, 1.0322]),
    30: ('gated', ['J1(x,8)', 'x[:,9]', 's_all(x,1)-s_all(x,2)'], 'pair30(x)', [0.4971, -0.2389, 0.2266, -0.0519]),
    31: ('relu', ['x[:,9]', 'topk(x,9,1)'], None, [5.3548, -5.0504, -0.1253]),
}
LABELS = {
    0: ('B', 'Fires when x2 is among the two largest of x0..x8 (strongest if it is the largest), growing with x2; also a smaller boost when x0 is the largest of x0..x8.'),
    1: ('B', 'Positive when x9 lies below a level set by the two largest of x0..x8 (about 1.4*second + 0.4*max - 1.9*x9); level raised when the largest of x0..x8 is x0, x2 or x3.'),
    2: ('B', 'Positive when x9 lies below a mix of the largest and second-largest of x0..x8 (about 1.2*max + 0.9*second - 2*x9); switched off when x8 is the largest of x0..x8.'),
    3: ('A', 'Silent if x8 is among the two largest of x0..x8; otherwise grows as x8 falls below the second-largest of x0..x8 (about 1.33*second - 1.45*x8), a bit higher if the largest is x5 or x6.'),
    4: ('B', 'Fires when x4 is among the two largest of x0..x8 (more if it is the largest), growing with x4; a smaller boost when the largest of x0..x8 is x1 or x2.'),
    5: ('A', 'Always on: weighted sum of the largest and second-largest of x0..x8 (about 3.3*max + 2.1*second), plus a small bump when the largest of x0..x8 is x1, x5 or x7.'),
    6: ('A', 'How far x9 falls below the second-largest of x0..x8: about 3*(second - x9) (+0.2*max), rectified; zero when x9 is among the two largest of all ten inputs.'),
    7: ('A', 'Fires when x6 is among the two largest of x0..x8 (more if it is the largest), growing with x6 and decreasing with the sum of the other inputs x0..x8.'),
    8: ('B', 'Positive when x9 lies below a level set by the top two of x0..x8 (about 0.8*max + 1.1*second - 1.9*x9); level raised when the largest of x0..x8 is at an even position (x0,x2,x4,x6,x8).'),
    9: ('A', 'Amount by which x8 exceeds all earlier inputs: about 7.25*ReLU(x8 - max(x0..x7)). Does not depend on x9.'),
    10: ('A', 'Silent when x8 is among the two largest of x0..x8; otherwise a moderate level that rises with the second-largest of x0..x7 and drops the later the largest of x0..x8 occurs.'),
    11: ('D', 'Poorly understood, low-variance: rises when x8 is the second-largest of x0..x8 (exceeds second-largest of others but not the max) and when x2 is the largest; reduced when x8 is the largest.'),
    12: ('A', 'Always on: weighted sum of the largest and second-largest of x0..x8 (about 5.7*max + 3.5*second), plus a small bump when the largest of x0..x8 is x0, x2, x3, x4 or x6.'),
    13: ('A', 'Fires when x5 is among the two largest of x0..x8, growing with x5 and with how far x5 exceeds the second-largest of the other values in x0..x8.'),
    14: ('B', 'Active only when the largest of x0..x8 sits at x0, x1, x2, x3 or x5; then about 0.8*max - 0.37*second of x0..x8 + 0.3, with an extra boost when x3 is the largest.'),
    15: ('B', 'Positive when x9 lies below a level set by the top two of x0..x8 (about 0.5*max + 1.4*second - 2*x9); level raised when the largest of x0..x8 is x1 or x4.'),
    16: ('A', 'Fires only when x4 is the largest of x0..x8; then about 0.9*x4 - 0.11*(sum of the other inputs x0..x8) + 0.15.'),
    17: ('D', 'Poorly understood position mix: on when x6 is the overall largest or x7 is second-largest of x0..x8; silenced when x8 is among the top two or x7/x8 is the largest.'),
    18: ('A', 'x9 rising above the second-largest of x0..x8 (about 4*x9 - 3.2*second - 0.6*max), rectified; only when the largest of x0..x8 is in x0..x6 (off if it is x7 or x8).'),
    19: ('A', 'x9 rising above a mix of the top two of x0..x8 (about 4.5*x9 - 2.9*second - 1.6*max), rectified; slightly higher when the largest of x0..x8 is x2, x3, x5, x6 or x8.'),
    20: ('B', 'Positive when x9 lies below a level set by the top two of x0..x8 (about 0.85*max + 1.0*second - 1.8*x9); switched off when x8 is the largest of x0..x8.'),
    21: ('D', 'Poorly understood, always on: roughly decreases with the sum of x0..x7 and increases with the three largest values of x0..x8.'),
    22: ('A', 'Amount by which x9 exceeds all earlier inputs: about 6.5*ReLU(x9 - max(x0..x8)), i.e. x9 is the overall maximum.'),
    23: ('A', 'Fires only when x7 is the largest of x0..x8; then about 1.07*x7 - 0.14*(sum of the other inputs x0..x8).'),
    24: ('A', 'Silent if x7 is among the two largest of x0..x8; otherwise grows as x7 falls below the second-largest of the other values in x0..x8 (about 2.1*second - 1.2*x7).'),
    25: ('B', 'Fires when x1 or x3 is among the two largest of x0..x8, growing with how far each exceeds the second-largest of the other values; strongest when x1 is the largest.'),
    26: ('A', 'Gap between the largest and second-largest of x0..x8, about 6.7*(max - second), rectified; zero when x8 itself is the largest of x0..x8.'),
    27: ('A', 'Fires when x0 is among the two largest of x0..x8 (more if it is the largest), growing with x0 and decreasing with the sum of the other inputs x0..x8.'),
    28: ('A', 'Silent if x3 is among the two largest of x0..x8; otherwise grows with the top two values of x0..x8 and as x3 decreases; lower when the largest of x0..x8 is x1 or x5..x8.'),
    29: ('B', 'High when x1 is NOT among the two largest of x0..x8: drops steeply as x1 exceeds the second-largest of the others, rises with max(x0..x3), and with x0 exceeding the second-largest of others.'),
    30: ('A', 'Fires only when the two largest of all ten inputs are x8 and one of x3, x5, x6, x7; larger when x8 is the largest and with a bigger second-to-third gap; reduced by large x9.'),
    31: ('A', 'Amount by which x9 exceeds the second-largest of x0..x8: about ReLU(5.36*x9 - 5.05*second - 0.13); positive means x9 is among the two largest of all ten inputs.'),
}
CONSTANTS = {i: len(SPEC[i][3]) for i in range(32)}
STRUCT = {
    0: "form=relu; features=['x[:,2]', 'I1(x,2)', 'I2(x,2)', 'inset(am9(x),[0])']",
    1: "form=relu; features=['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[0, 2, 3])']",
    2: "form=relu; features=['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[8])']",
    3: "form=gated; features=['x[:,8]', 'topk(x,9,0)', 'topk(x,9,1)', 'inset(am9(x),[5,6])']; gate=1-top2(x,8)",
    4: "form=relu; features=['x[:,4]', 'I1(x,4)', 'I2(x,4)', 'inset(am9(x),[1,2])']",
    5: "form=relu; features=['topk(x,9,0)', 'topk(x,9,1)', 'inset(am9(x),[1,5,7])']",
    6: "form=relu; features=['topk(x,9,1)', 'topk(x,9,0)', 'x[:,9]']",
    7: "form=relu; features=['x[:,6]', 'so(x,6)', 'I1(x,6)', 'I2(x,6)']",
    8: "form=relu; features=['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[0, 2, 4, 6, 8])']",
    9: "form=scale; features=['relu(x[:,8]-topk(x,8,0))']",
    10: "form=gated; features=['x[:,8]', 'So9(x,8)', 'am9(x)', 'Mo9(x,8)']; gate=1-top2(x,8)",
    11: "form=relu; features=['rSo9(x,8)', 'I1(x,2)', 'rMo9(x,8)', 'topk(x,6,0)']",
    12: "form=relu; features=['topk(x,9,0)', 'topk(x,9,1)', 'inset(am9(x),[0,2,3,4,6])']",
    13: "form=relu; features=['x[:,5]', 'I1(x,5)', 'I2(x,5)', 'rSo9(x,5)']",
    14: "form=gated; features=['topk(x,9,0)', 'topk(x,9,1)', 'I1(x,3)']; gate=inset(am9(x),[0,1,2,3,5])",
    15: "form=relu; features=['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[1, 4])']",
    16: "form=gated; features=['x[:,4]', 'so(x,4)']; gate=I1(x,4)",
    17: "form=relu; features=['rSo9(x,8)', 'J1(x,6)', 'I2(x,7)', 'topk(x,5,0)']",
    18: "form=relu; features=['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[0, 1, 2, 3, 4, 5, 6])']",
    19: "form=relu; features=['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[2, 3, 5, 6, 8])']",
    20: "form=relu; features=['topk(x,9,0)', 'topk(x,9,1)', 'x[:,9]', 'inset(am9(x),[0, 1, 2, 3, 4, 5, 6, 7])']",
    21: "form=lin; features=['x[:,:8].sum(1)', 'topk(x,9,0)', 'topk(x,9,1)', 'topk(x,9,2)']",
    22: "form=scale; features=['relu(x[:,9]-topk(x,9,0))']",
    23: "form=gated; features=['x[:,7]', 'so(x,7)']; gate=I1(x,7)",
    24: "form=gated; features=['x[:,7]', 'So9(x,7)', 'so(x,7)']; gate=1-top2(x,7)",
    25: "form=relu; features=['rSo9(x,1)', 'rSo9(x,3)', 'I1(x,1)', 'I2(x,1)']",
    26: "form=relu; features=['topk(x,8,0)', 'topk(x,9,1)']",
    27: "form=relu; features=['x[:,0]', 'so(x,0)', 'I1(x,0)', 'I2(x,0)']",
    28: "form=gated; features=['x[:,3]', 'topk(x,9,0)', 'topk(x,9,1)', 'inset(am9(x),[1,5,6,7,8])']; gate=1-top2(x,3)",
    29: "form=relu; features=['rSo9(x,1)', 'topk(x,4,0)', 'x[:,1]', 'rSo9(x,0)']",
    30: "form=gated; features=['J1(x,8)', 'x[:,9]', 's_all(x,1)-s_all(x,2)']; gate=pair30(x)",
    31: "form=relu; features=['x[:,9]', 'topk(x,9,1)']",
}

def f(i, x):
    x = np.asarray(x, dtype=float)
    kind, feats, gate, c = SPEC[i]
    F = np.column_stack([eval(e, globals(), {'x': x}) for e in feats])
    c = np.asarray(c)
    if kind == 'scale':
        return c[0] * F[:, 0]
    z = F @ c[:-1] + c[-1]
    if kind == 'lin':
        return z
    p = relu(z)
    if gate is not None:
        p = p * (eval(gate, globals(), {'x': x}) > 0.5)
    return p
