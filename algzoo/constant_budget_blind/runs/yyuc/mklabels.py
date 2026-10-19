import json
fit=json.load(open('fit.json')); SPEC={int(k):v for k,v in fit['SPEC'].items()}; C={int(k):v for k,v in fit['C'].items()}
L={
0:("A","Fires when an earlier value is high but x9 is low: ReLU(~13*(max of x0..x8, taken as 0 if x8 holds it) - 7.6*x9), weaker when that max sits at x1,x4,x5,x7, strongest at x0,x2,x3,x6."),
1:("B","Large when the top of x0..x7 sits at x2, x5 or x6 and exceeds the rest: ReLU(25*max(x0..x7) - ~4-6 times each of max(x0,x1,x3,x4,x7), max(x0,x4,x7), max(x0..x8 except x2,x5), second largest of x0..x8 except x2)."),
2:("D","Weak, poorly understood: roughly ReLU(8*(second largest of x0,x1,x2,x3,x4,x6,x7) + 4*x2 - 4*(second largest of x0..x8 unless at x2) + small terms in max(x1,x4) and x2 when third largest)."),
3:("A","Fires when an earlier value is high but x9 is low: ReLU(~12.7*(max of x0..x8, taken as 0 if x8 holds it) - 8*x9), weaker when that max sits at x0,x3,x6,x7, strongest at x1,x2,x4,x5."),
4:("A","Margin of the largest over the second largest of x0..x8: ReLU(~16.6*largest - ~16*second + 5.4*max(x0,x1,x3,x4,x6,x7)); zero when x8 is the largest, smaller when the second largest is x8."),
5:("C","Mostly ~8*max(x0,x2,x3,x5,x6,x7,x8), plus ~4*largest of x0..x8 when it is at x0,x2,x3,x5,x6,x8; strongly reduced when the largest or second largest of x0..x8 is x5. ReLU applied."),
6:("A","Fires when an earlier value is high but x9 is low: ReLU(~12.4*(max of x0..x8, taken as 0 if x8 holds it) - 8*x9), weaker when that max sits at x2,x3,x5,x6, strongest at x0,x1,x4,x7."),
7:("A","Stores the running maximum: ~17*max(x0..x7) + 5*max(x0,x3,x6) + 7*max(x0,x2,x3,x5,x6,x7,x8) + 2.6*(max of x0..x8 if at x3 or x6); strongest when the maximum sits at x0,x3,x6."),
8:("B","ReLU(13*max(x0,x1,x3,x4,x7) - 7*second largest of x0..x7) plus bonuses (~2-4 times the largest of x0..x8) when that largest sits at x4 or x7; bonuses missing when it sits at x2 or x3."),
9:("C","ReLU(6.4*max(x0,x1,x4,x7) + 3*x1 + ~3.4*largest of x0..x8 if at x0,x1,x3,x4,x7 + ~1.7*that largest if not at x0,x3,x6 + 3.8*second largest of x0..x8 if it is x1)."),
10:("A","Last input beats the earlier runner-up: ReLU(8.4*x9 - 4.7*P2 - 1.9*P3), P2,P3 = second and third largest of x0..x8. Positive essentially when x9 is among the two largest inputs."),
11:("C","ReLU(8*(largest of x0..x8 if at x1,x2,x3,x5,x6,x8) + 5.2*(that largest if at x0,x2,x4,x5,x8) - 5.3*(second largest of x0..x8 unless at x8) - 5.1*(second largest if at x7))."),
12:("A","Margin of the largest over the second largest of x0..x8: ReLU(21*largest - 12.4*second - 2.3*max(x2,x5) - 3*max(x1,x2,x5,x6,x8)); smaller when the big values sit at x2,x5."),
13:("D","Partly understood: ReLU(10.4*max(x0,x1,x3,x4,x7) + 7.6*second largest of x0,x1,x3,x4,x7 - 2.6*max(x4,x7) + 1.4*x2 - 2.8*(second largest of x0..x8 if at x4 or x7))."),
14:("A","Maximum read with a position-dependent weight: ReLU(10*max(x1,x2,x4,x5,x7,x8) + 15.5*max(x0,x1,x2,x3,x5,x6,x8) - 5.3*max(x0,x2,x3,x6) + 4.5*(max of x0..x8 if at x1,x2,x4,x5,x7,x8))."),
15:("A","New maximum at the very end: 26.8*ReLU(x9 - 0.989*max(x0..x8)); positive only when x9 is the largest of all ten inputs, growing with its lead."),
}
for i,(c,t) in L.items(): assert len(t.split())<=40,(i,len(t.split()))
src=open('featlib_src.py').read()
out='''"""Neuron labels and formulas for h_10 of the second-largest RNN.
Every formula is f = ReLU(sum_k c_k * feature_k) over interpretable features (see _feat)."""
import numpy as np

LABELS = %s

SPEC = %s

COEF = %s

CONSTANTS = {i: len(COEF[i]) for i in range(16)}

STRUCT = {i: "searched: feature list " + ", ".join(SPEC[i]) + " (position sets chosen by greedy search, mostly arcs of the phase circle 2,5,8,1,4,7,0,3,6)" for i in range(16)}

%s

def f(i, x):
    x = np.asarray(x, dtype=float)
    z = np.zeros(len(x))
    for name, c in zip(SPEC[i], COEF[i]):
        z = z + c * _feat(name, x)
    return np.maximum(z, 0.0)
'''%(repr(L),repr(SPEC),repr(C),src)
open('labels.py','w').write(out)
