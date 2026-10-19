import json, re
out=json.load(open('c2_out.json'))
src=open('featlib.py').read()
src=src.replace('def feature(','def _feature(').replace('def design(x, names):\n    return np.stack([feature(x, n) for n in names], 1)','')
L={
0:("A","Fires when x9 falls well below the largest of x0..x8: grows with that maximum (weight about 2x larger if it sits at x0,x2,x3,x5,x6 than at x1,x4,x7) and the second-largest, minus about 7.7*x9; silent if x8 is largest of x0..x8."),
1:("C","Mixture of maxima over position groups: max(x1,x2,x3,x5,x6) + max(x0,x2..x7) + max(x0..x8) minus max(x0,x3,x4,x6,x7,x8) and second-largest of x0,x1,x3..x7; strongest when the overall maximum is at x2 or x5."),
2:("D","Not well understood. Roughly the second-largest of x0..x4 plus a little of x2 itself; strongest when x2 is the largest or second-largest input, small otherwise."),
3:("A","Fires when x9 falls well below the largest of x0..x8: grows with that maximum (weight about 2x larger if it sits at x1,x2,x4,x5 than at x0,x3,x6,x7) and the second-largest, minus about 8.2*x9; silent if x8 is largest of x0..x8."),
4:("A","Margin of the maximum: largest of x0..x7 minus the second-largest of x0..x8 (so x8 being largest suppresses it), slightly lowered by large values at x2,x5,x8 and by x8 being second-largest overall."),
5:("A","Position-weighted copy of the maximum of x0..x8: large when the largest value is at x0,x2,x3,x6,x7 or x8, weak when at x1,x4,x5; lowered by the second-largest of x0,x1,x3,x4,x5,x6; x9 ignored."),
6:("A","Fires when x9 falls well below the largest of x0..x8: grows with that maximum (weight about 2x larger if it sits at x0,x1,x4,x7 than at x2,x3,x5,x6) and the second-largest, minus about 8.2*x9; silent if x8 is largest of x0..x8."),
7:("A","Size of the running maximum: mainly the largest of x0..x7, with extra weight when the largest value sits at x0,x3,x6 (also x2,x7,x8); almost always positive and ignores x9."),
8:("A","Position-weighted maximum: mainly max(x0,x1,x3,x4,x7), minus the second-largest of x0..x5, plus max(x0,x1,x4,x5,x6,x7) corrections; small when the largest value sits at x2, x5 or x9."),
9:("B","Largest value among x0,x1,x4,x7 plus extra direct weight on x1 and on x4 when x4 sets a new running record; reduced by the second-largest of x0,x4,x5,x6,x7. Small when the maximum is at x2,x3,x5,x6,x8,x9."),
10:("A","Signals that the last input x9 is among the top two: about 8.3*x9 minus 4.6*(second-largest of x0..x8) minus 2.0*(third-largest of x0..x8), floored at zero."),
11:("C","Largest of x2,x5,x6,x8 plus largest of x1..x6,x8, minus the second-largest of x0..x7 and minus maxima over x4,x6,x7 and x1,x3,x4,x7; essentially silent whenever x7 is the largest input."),
12:("A","Margin of the maximum: largest of x0..x8 minus the second- and third-largest of x0..x8, slightly lowered by large values at x1,x2,x5,x6,x8; x9 ignored."),
13:("B","Size of early large values: largest of x0,x1,x3,x4,x7 plus the second-largest of x0..x3 and of x0..x4, minus the second-largest of x0,x2,x4,x6; favours maxima at x0,x1,x3,x7."),
14:("A","Position-weighted copy of the maximum of x0..x8: large when the largest value sits at x1,x2,x4,x5,x7 or x8, weaker when at x0,x3,x6; x9 ignored."),
15:("A","How much the last input x9 exceeds every earlier input: about 27*max(0, x9 - max(x0..x8)); zero unless x9 is the overall maximum."),
}
names={i:out[str(i)][0] for i in range(16)}; W={i:[round(v,4) for v in out[str(i)][1]] for i in range(16)}
struct={i:"features searched greedily from a dictionary (values, sorted/running maxima & second maxima, maxima/second maxima over position subsets, rank-at-position gates); chosen: "+"; ".join(names[i]) for i in range(16)}
with open('labels.py','w') as f:
    f.write('"""Per-neuron labels and formulas for h_10 of the second-argmax RNN.\nFeature names: xj = input j; a1/a2/a3 = 1st/2nd/3rd largest of x0..x8; a1@{S} = a1 if its position is in S;\nsuffix g = multiplied by [largest of x0..x8 is not at x8]; s2@j = overall 2nd largest if at j;\nPt/Rt/Tt = largest/2nd/3rd of x0..xt; Mc = max(xc, xc+3, ...); max{S}/sec{S} = largest/2nd largest over positions S;\nnewj = max(0, xj - max(x0..x(j-1))); up9 = max(0, x9 - a1). Each formula: h = max(0, sum_k c_k * feature_k)."""\n')
    f.write('import numpy as np\n\n'+src+'\n\n')
    f.write('LABELS = '+repr(L)+'\n')
    f.write('FEATURES = '+repr(names)+'\n')
    f.write('COEFS = '+repr(W)+'\n')
    f.write('CONSTANTS = '+repr({i:len(W[i]) for i in range(16)})+'\n')
    f.write('STRUCT = '+repr(struct)+'\n\n')
    f.write('def f(i, x):\n    x = np.asarray(x, dtype=float)\n    F = np.stack([_feature(x, n) for n in FEATURES[i]], 1)\n    return np.maximum(F @ np.array(COEFS[i]), 0.0)\n')
for i,(c,l) in L.items(): print(i,len(l.split()))
