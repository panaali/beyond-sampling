"""Per-neuron labels and formulas for h_10 of the second-largest-position RNN.
All formulas have the form ReLU(sum_k c_k * feature_k(x)); features are defined in feat().
Prefix = x0..x8; M1>=M2>=M3 its top values; am/as = positions of its largest/second-largest."""
import numpy as np
import math
def _r(a): return np.maximum(a,0)
def feat(name, x):
    """Evaluate a named feature. Prefix = x0..x8. Q1>=Q2>=Q3 its top values, am/as = positions of its largest/second."""
    p=x[:,:9]; Q=-np.sort(-p,1); s=np.argsort(-p,1)
    Q1,Q2,Q3=Q[:,0],Q[:,1],Q[:,2]; am,as_=s[:,0],s[:,1]
    t=name.split('|')
    k=t[0]
    if k=='Q1': return Q1
    if k=='Q2': return Q2
    if k=='Q3': return Q3
    if k=='x9': return x[:,9]
    if k=='x': return x[:,int(t[1])]
    if k=='rnew8': return _r(x[:,8]-x[:,:8].max(1))
    if k=='gap': return Q1-Q2
    if k=='rxQ3': j=int(t[1]); return _r(x[:,j]-Q3)
    if k=='rxQ2': j=int(t[1]); return _r(x[:,j]-Q2)
    if k in ('rx8S7','rS7x8','rP7x8'):
        f8=-np.sort(-x[:,:8],1); P7,S7=f8[:,0],f8[:,1]; x8=x[:,8]
        return {'rx8S7':_r(x8-S7),'rS7x8':_r(S7-x8),'rP7x8':_r(P7-x8)}[k]
    if k=='rx': return _r(x[:,int(t[1])])
    if k=='rQ2x': j=int(t[1]); return _r(Q2-x[:,j])
    if k=='rQ1x': j=int(t[1]); return _r(Q1-x[:,j])
    S=[int(c) for c in t[2]]
    if k=='am': ind=np.isin(am,S)
    elif k=='as': ind=np.isin(as_,S)
    elif k=='top2': ind=np.isin(am,S)|np.isin(as_,S)
    v=t[1]
    base={'Q1':Q1,'Q2':Q2,'Q3':Q3,'g12':Q1-Q2,'g23':Q2-Q3}[v]
    return ind*base


SPEC = {
    0: (['x9', 'am|Q1|1256', 'Q2', 'am|Q1|8', 'as|Q1|0378'], [-1.5492, 0.5987, 1.2111, -3.7487, 0.2902]),
    1: (['gap', 'rnew8', 'Q1'], [4.1787, -8.8011, 0.3275]),
    2: (['x|3', 'x|5', 'rxQ3|0', 'as|Q1|3', 'top2|Q1|3'], [0.7436, -0.2205, 0.9489, -1.005, 2.4014]),
    3: (['x9', 'rxQ2|1', 'top2|Q1|8', 'am|g12|0234', 'as|Q2|567'], [-0.2627, 1.1073, -1.0603, 1.6759, 0.9828]),
    4: (['x9', 'rxQ2|1', 'rQ1x|7', 'as|Q2|0246', 'as|g23|0468'], [-0.1685, 0.4672, 0.1347, 0.6304, 0.4747]),
    5: (['gap', 'rQ1x|2', 'am|Q1|0348', 'top2|Q2|014', 'am|g12|18'], [2.7995, 0.419, 0.7528, 0.4944, 1.3005]),
    6: (['x9', 'Q1', 'am|g12|1368', 'am|Q1|02', 'as|Q2|1346'], [-1.8682, 1.3934, -1.0507, 0.504, 0.418]),
    7: (['x9', 'Q1'], [8.7541, -8.6804]),
    8: (['Q1', 'x|6', 'x|7', 'rQ2x|4', 'am|Q1|0125'], [1.4477, -0.4154, -0.5015, 0.5326, 0.9012]),
    9: (['Q2', 'x9', 'rQ1x|0', 'top2|Q2|08', 'as|g23|0248'], [-3.2715, -0.2335, -0.0367, -0.8182, 1.2568]),
    10: (['x9', 'am|Q1|45', 'as|Q2|58', 'top2|Q1|35', 'top2|Q1|678'], [-0.2175, 0.6368, -0.4189, -0.1387, -0.3186]),
    11: (['rQ1x|8', 'am|Q1|4', 'am|Q1|046', 'top2|Q2|46', 'am|Q1|3578'], [0.1475, 0.6989, 0.7188, -0.3941, -0.2841]),
    12: (['x9', 'Q2'], [-3.0631, 3.1568]),
    13: (['Q1', 'x9', 'am|Q1|8', 'am|Q2|1247', 'am|g12|0358'], [1.331, -1.6818, -4.207, 0.6958, -0.8415]),
    14: (['rQ2x|5', 'rQ2x|8', 'rQ1x|4', 'top2|Q1|8', 'am|g12|0124'], [0.4358, -3.6325, 0.5117, 0.3505, 1.1106]),
    15: (['rxQ3|1', 'rQ1x|6', 'am|Q1|1', 'as|Q1|0', 'top2|Q1|1'], [1.3316, 0.126, 1.2823, 0.3407, 1.2028]),
    16: (['x9', 'Q2'], [5.6096, -5.4451]),
    17: (['Q1', 'Q2'], [8.1185, 2.0343]),
    18: (['Q1', 'x9', 'am|Q1|1', 'am|g12|2357', 'as|Q2|2'], [1.0421, -1.4327, 0.9642, -1.0117, 0.5063]),
    19: (['rQ2x|7', 'am|Q1|6', 'am|Q1|026', 'as|Q2|26', 'top2|Q1|6'], [0.1071, 0.904, 0.2032, 0.3137, 1.1958]),
    20: (['Q2', 'rxQ2|1', 'rQ1x|0', 'top2|Q2|18', 'top2|Q1|0'], [0.7037, 1.0874, 0.5117, 0.4092, -0.6187]),
    21: (['x9', 'Q1', 'Q2', 'am|g12|1357'], [4.2674, -1.4776, -2.7117, -2.1363]),
    22: (['top2|Q1|7', 'am|Q1|7'], [0.6214, 1.9162]),
    23: (['x9', 'Q2', 'rnew8', 'Q1', 'am|Q1|2678'], [4.7345, -3.7109, -25.9656, -0.7423, -0.5258]),
    24: (['Q2', 'x|5', 'rQ1x|6', 'rQ1x|8', 'top2|Q2|5'], [1.1175, -0.4378, 0.191, 0.183, -2.3199]),
    25: (['Q2', 'rQ1x|8', 'as|Q2|5678', 'top2|Q1|8', 'top2|Q2|78'], [1.0237, 0.2485, -0.3386, -0.6358, -0.2452]),
    26: (['rxQ3|2', 'rQ1x|6', 'top2|Q1|027', 'am|Q1|024', 'top2|Q1|2'], [1.8272, 0.2785, 0.4064, 1.2246, 0.8107]),
    27: (['am|Q1|26', 'am|Q2|26'], [1.5829, -0.6197]),
    28: (['Q1', 'rQ2x|8', 'am|g12|678', 'am|Q1|013', 'am|Q1|8'], [0.7779, 0.2289, -0.5586, 0.3523, -0.7433]),
    29: (['Q1', 'x9', 'am|Q1|8', 'am|Q2|0367', 'am|Q1|3'], [0.9988, -1.6309, -3.8367, 0.8632, 0.5439]),
    30: (['Q2', 'rQ2x|1', 'rQ1x|4', 'top2|Q1|4', 'am|g12|03'], [1.0442, 0.1543, 0.5974, -0.9982, 0.7714]),
    31: (['Q2', 'am|g12|0134', 'x|2', 'Q1', 'am|Q2|0134'], [5.4905, 0.9231, -0.3146, 0.6056, 0.4956]),
}
LABELS = {
    0: ('A', 'x9-is-low detector, off when x8 is the max: ReLU(1.21·M2 − 1.55·x9 + 0.6·M1·[max at x1,x2,x5,x6] + 0.29·M1·[second at x0,x3,x7,x8] − 3.75·M1·[max at x8]); M1,M2 = largest, second-largest of x0..x8.'),
    1: ('A', 'Lead of the running maximum: ReLU(4.18·(M1−M2) + 0.33·M1 − 8.8·ReLU(x8 − max(x0..x7))); M1,M2 = largest, second-largest of x0..x8. Gap between top two, silenced when x8 is the new maximum.'),
    2: ('A', 'x3-is-a-top-value detector: ≈2.4·M1 when x3 is among the two largest of x0..x8 (M1 = largest), minus M1 if x3 is only second, plus 0.74·x3 − 0.22·x5 + 0.95·ReLU(x0 − third-largest).'),
    3: ('C', 'Mixed: ReLU(1.68·(M1−M2)·[max at x0,x2,x3,x4] + 0.98·M2·[second at x5,x6,x7] + 1.11·ReLU(x1−M2) − 1.06·M1·[x8 in top two] − 0.26·x9); M1,M2 = largest, second-largest of x0..x8.'),
    4: ('D', 'Weak, unclear: ReLU(0.63·M2·[second at x0,x2,x4,x6] + 0.48·(M2−M3)·[second at x0,x4,x6,x8] + 0.47·ReLU(x1−M2) + 0.14·ReLU(M1−x7) − 0.17·x9); M1≥M2≥M3 = top three of x0..x8.'),
    5: ('B', 'Lead of the maximum plus position bonuses: ReLU(2.8·(M1−M2) + 1.3·(M1−M2)·[max at x1,x8] + 0.75·M1·[max at x0,x3,x4,x8] + 0.49·M2·[x0,x1,x4 in top two] + 0.42·ReLU(M1−x2)); M1,M2 = largest, second-largest of x0..x8.'),
    6: ('B', 'x9-is-low detector with max-position code: ReLU(1.39·M1 − 1.87·x9 − 1.05·(M1−M2)·[max at x1,x3,x6,x8] + 0.5·M1·[max at x0,x2] + 0.42·M2·[second at x1,x3,x4,x6]); M1,M2 = largest, second-largest of x0..x8.'),
    7: ('A', 'Fires when x9 is a new maximum: ReLU(8.75·x9 − 8.68·max(x0..x8)), i.e. about 8.7 times the amount by which x9 exceeds all earlier inputs.'),
    8: ('D', 'Unclear, always positive: ≈1.45·M1 + 0.9·M1·[max of x0..x8 at x0,x1,x2,x5] + 0.53·ReLU(M2 − x4) − 0.42·x6 − 0.5·x7; M1,M2 = largest, second-largest of x0..x8.'),
    9: ('A', 'All-values-low detector, rarely active: ReLU(−3.27·M2 − 0.23·x9 − 0.82·M2·[x0 or x8 in top two] + 1.26·(M2−M3)·[second at x0,x2,x4,x8] − 0.04·ReLU(M1−x0)); M1≥M2≥M3 = top three of x0..x8.'),
    10: ('C', 'Mixed max-position code: ReLU(0.64·M1·[max at x4,x5] − 0.42·M2·[second at x5,x8] − 0.14·M1·[x3 or x5 in top two] − 0.32·M1·[x6,x7 or x8 in top two] − 0.22·x9); M1,M2 = largest, second-largest of x0..x8.'),
    11: ('A', 'Max-position code (max at x0, x4, x6): ReLU(0.72·M1·[max at x0,x4,x6] + 0.7·M1·[max at x4] − 0.28·M1·[max at x3,x5,x7,x8] − 0.39·M2·[x4 or x6 in top two] + 0.15·ReLU(M1−x8)); M1,M2 = largest, second-largest of x0..x8.'),
    12: ('A', 'Fires when x9 is below the second-largest of x0..x8 (call it M2): ReLU(3.16·M2 − 3.06·x9), i.e. x9 is not among the top two so far.'),
    13: ('A', 'x9-is-low detector, off when x8 is the max: ReLU(1.33·M1 − 1.68·x9 − 4.21·M1·[max at x8] + 0.7·M2·[max at x1,x2,x4,x7] − 0.84·(M1−M2)·[max at x0,x3,x5,x8]); M1,M2 = largest, second-largest of x0..x8.'),
    14: ('C', 'Mostly x8 relative to the top: ReLU(−3.63·ReLU(M2−x8) + 0.35·M1·[x8 in top two] + 1.11·(M1−M2)·[max at x0,x1,x2,x4] + 0.44·ReLU(M2−x5) + 0.51·ReLU(M1−x4)); M1,M2 = largest, second-largest of x0..x8.'),
    15: ('A', 'x1-is-a-top-value detector: ReLU(1.2·M1·[x1 among two largest] + 1.28·M1·[x1 largest] + 1.33·ReLU(x1−M3) + 0.34·M1·[second at x0] + 0.13·ReLU(M1−x6)); M1, M3 = largest, third-largest of x0..x8.'),
    16: ('A', 'Fires when x9 exceeds the second-largest of x0..x8 (call it M2): ReLU(5.61·x9 − 5.45·M2), i.e. x9 enters the top two.'),
    17: ('A', 'Running top-two sum: ReLU(8.12·M1 + 2.03·M2), where M1, M2 are the largest and second-largest of x0..x8; x9 has no effect.'),
    18: ('B', 'x9-is-low detector with max-position code: ReLU(1.04·M1 − 1.43·x9 + 0.96·M1·[max at x1] − 1.01·(M1−M2)·[max at x2,x3,x5,x7] + 0.51·M2·[second at x2]); M1,M2 = largest, second-largest of x0..x8.'),
    19: ('A', 'x6-is-a-top-value detector: ReLU(1.2·M1·[x6 among two largest] + 0.9·M1·[x6 largest] + 0.2·M1·[max at x0,x2,x6] + 0.31·M2·[second at x2,x6] + 0.11·ReLU(M2−x7)); M1,M2 = largest, second-largest of x0..x8.'),
    20: ('D', "Unclear, roughly 'x0 is not a top value': ReLU(0.7·M2 + 1.09·ReLU(x1−M2) + 0.51·ReLU(M1−x0) + 0.41·M2·[x1 or x8 in top two] − 0.62·M1·[x0 in top two]); M1,M2 = largest, second-largest of x0..x8."),
    21: ('B', 'Fires when x9 is large versus x0..x8: ReLU(4.27·x9 − 1.48·M1 − 2.71·M2 − 2.14·(M1−M2)·[max at odd position x1,x3,x5,x7]); M1,M2 = largest, second-largest of x0..x8.'),
    22: ('A', 'x7-is-the-max detector: 1.92·M1·[x7 is the largest of x0..x8] + 0.62·M1·[x7 is among the two largest], with M1 = largest of x0..x8.'),
    23: ('A', 'Fires when x9 enters the top two while x8 is not the max: ReLU(4.73·x9 − 3.71·M2 − 0.74·M1 − 26·ReLU(x8 − max(x0..x7)) − 0.53·M1·[max at x2,x6,x7,x8]); M1,M2 = largest, second-largest of x0..x8.'),
    24: ('C', "Roughly 'x5 is not a top value': ReLU(1.12·M2 − 0.44·x5 − 2.32·M2·[x5 among two largest] + 0.19·ReLU(M1−x6) + 0.18·ReLU(M1−x8)); M1,M2 = largest, second-largest of x0..x8."),
    25: ('B', 'Second-largest tracker damped when late inputs are top: ReLU(1.02·M2 + 0.25·ReLU(M1−x8) − 0.64·M1·[x8 in top two] − 0.34·M2·[second at x5..x8] − 0.25·M2·[x7 or x8 in top two]); M1,M2 = largest, second-largest of x0..x8.'),
    26: ('A', 'x2/x0-is-a-top-value detector: ReLU(1.83·ReLU(x2−M3) + 1.23·M1·[max at x0,x2,x4] + 0.81·M1·[x2 in top two] + 0.41·M1·[x0,x2 or x7 in top two] + 0.28·ReLU(M1−x6)); M1, M3 = largest, third-largest of x0..x8.'),
    27: ('A', 'Max-at-x2-or-x6 detector: ReLU(1.58·M1 − 0.62·M2) when the largest of x0..x8 sits at x2 or x6, else 0; M1,M2 = largest, second-largest of x0..x8.'),
    28: ('D', 'Unclear max-position code: ReLU(0.78·M1 − 0.74·M1·[max at x8] − 0.56·(M1−M2)·[max at x6,x7,x8] + 0.35·M1·[max at x0,x1,x3] + 0.23·ReLU(M2−x8)); M1,M2 = largest, second-largest of x0..x8.'),
    29: ('A', 'x9-is-low detector, off when x8 is the max: ReLU(1.0·M1 − 1.63·x9 − 3.84·M1·[max at x8] + 0.86·M2·[max at x0,x3,x6,x7] + 0.54·M1·[max at x3]); M1,M2 = largest, second-largest of x0..x8.'),
    30: ('C', "Roughly 'x4 is not a top value': ReLU(1.04·M2 + 0.6·ReLU(M1−x4) − 1.0·M1·[x4 among two largest] + 0.77·(M1−M2)·[max at x0,x3] + 0.15·ReLU(M2−x1)); M1,M2 = largest, second-largest of x0..x8."),
    31: ('A', 'Second-largest tracker: ReLU(5.49·M2 + 0.61·M1 + 0.92·(M1−M2)·[max at x0,x1,x3,x4] + 0.5·M2·[max at x0,x1,x3,x4] − 0.32·x2); M1,M2 = largest, second-largest of x0..x8.'),
}
CONSTANTS = {i: len(SPEC[i][1]) for i in SPEC}
STRUCT = {
    0: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, am|Q1|1256, Q2, am|Q1|8, as|Q1|0378; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    1: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): gap, rnew8, Q1',
    2: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x|3, x|5, rxQ3|0, as|Q1|3, top2|Q1|3; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    3: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, rxQ2|1, top2|Q1|8, am|g12|0234, as|Q2|567; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    4: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, rxQ2|1, rQ1x|7, as|Q2|0246, as|g23|0468; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    5: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): gap, rQ1x|2, am|Q1|0348, top2|Q2|014, am|g12|18; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    6: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, Q1, am|g12|1368, am|Q1|02, as|Q2|1346; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    7: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, Q1',
    8: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q1, x|6, x|7, rQ2x|4, am|Q1|0125; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    9: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q2, x9, rQ1x|0, top2|Q2|08, as|g23|0248; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    10: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, am|Q1|45, as|Q2|58, top2|Q1|35, top2|Q1|678; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    11: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): rQ1x|8, am|Q1|4, am|Q1|046, top2|Q2|46, am|Q1|3578; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    12: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, Q2',
    13: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q1, x9, am|Q1|8, am|Q2|1247, am|g12|0358; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    14: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): rQ2x|5, rQ2x|8, rQ1x|4, top2|Q1|8, am|g12|0124; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    15: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): rxQ3|1, rQ1x|6, am|Q1|1, as|Q1|0, top2|Q1|1; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    16: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, Q2',
    17: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q1, Q2',
    18: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q1, x9, am|Q1|1, am|g12|2357, as|Q2|2; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    19: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): rQ2x|7, am|Q1|6, am|Q1|026, as|Q2|26, top2|Q1|6; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    20: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q2, rxQ2|1, rQ1x|0, top2|Q2|18, top2|Q1|0; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    21: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, Q1, Q2, am|g12|1357; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    22: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): top2|Q1|7, am|Q1|7; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    23: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): x9, Q2, rnew8, Q1, am|Q1|2678; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    24: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q2, x|5, rQ1x|6, rQ1x|8, top2|Q2|5; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    25: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q2, rQ1x|8, as|Q2|5678, top2|Q1|8, top2|Q2|78; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    26: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): rxQ3|2, rQ1x|6, top2|Q1|027, am|Q1|024, top2|Q1|2; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    27: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): am|Q1|26, am|Q2|26; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    28: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q1, rQ2x|8, am|g12|678, am|Q1|013, am|Q1|8; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    29: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q1, x9, am|Q1|8, am|Q2|0367, am|Q1|3; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    30: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q2, rQ2x|1, rQ1x|4, top2|Q1|4, am|g12|03; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
    31: 'features chosen by greedy/beam search from a library (Q1,Q2,Q3,x9,x_j,gap,ReLU(x8-max x0..x7),ReLU(x_j-M2/M3),ReLU(M1/M2-x_j), position-set indicators): Q2, am|g12|0134, x|2, Q1, am|Q2|0134; position sets searched (derived from per-position coefficient tables, top/bottom-k)',
}

def f(i, x):
    x = np.asarray(x, dtype=float)
    names, co = SPEC[i]
    z = np.zeros(len(x))
    for n, c in zip(names, co):
        z = z + c * feat(n, x)
    return np.maximum(z, 0.0)
