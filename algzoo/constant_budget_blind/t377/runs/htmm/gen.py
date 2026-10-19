import json
sp=json.load(open('spec.json'))
D="Let P, Q, R be the largest, second- and third-largest of x0..x8. "
L={
0:("A","Fires when x2 is among the two largest of x0..x8: grows with x2 minus R (third-largest of x0..x8), with extra boosts when the largest sits at x2 or x0; small baseline from the total |x0|+..+|x6|."),
1:("B","Grows as x9 falls below Q, the second-largest of x0..x8 (x9 not a top-two value), plus a position code: raised when the largest of x0..x8 sits at x0, x2 or x3."),
2:("B","Grows as x9 falls below the largest of x0..x8 (call it P), scaled about 2:1.7; silenced when P sits at x8, and lowered when P sits at x2 or x4."),
3:("A","How far x8 lies below Q, the second-largest of x0..x8 (zero when x8 is a top-two value of x0..x8); shifted up when the largest of x0..x8 sits at x5 or x6 (less for x3, x7)."),
4:("A","Fires when x4 is among the two largest of x0..x8, growing with x4 minus R (third-largest of x0..x8); boosted when the largest sits at x1 or x2; small baseline from |x0|+..+|x6|."),
5:("A","Weighted magnitude of the top two of x0..x8: about 3.3*P + 2.2*Q (P, Q = largest and second-largest of x0..x8), with offsets coding whether P sits at x1/x7 or x3/x5."),
6:("A","How far x9 lies below Q, the second-largest of x0..x8: about 3.1*max(0, Q - x9). Zero when x9 would be among the two largest."),
7:("A","Fires when x6 is among the two largest of x0..x8, growing with x6 minus R (third-largest of x0..x8); reduced by Q (second-largest), small baseline from |x0|+..+|x7|."),
8:("B","Grows as x9 falls below Q, the second-largest of x0..x8, plus a position code: raised when the largest of x0..x8 sits at x4 or x6 (and x0 or x8)."),
9:("A","By how much x8 sets a new running maximum: about 7.15*max(0, x8 - max(x0..x7)). Ignores x9."),
10:("A","Signals that x8 is NOT among the two largest of x0..x8 (off when it is; weaker when x6/x7 are top-two), otherwise rising with Q (second-largest of x0..x8) and |x0|+..+|x6|."),
11:("D","Unclear position code: active mainly when the largest of x0..x8 sits at x2, x4 or x8, or the second-largest at x2 or x8; silent when the largest sits at x1, x3, x6 or x7."),
12:("A","Running magnitude of the top two values of x0..x8: about 5.76*P + 3.47*Q (P, Q = largest and second-largest of x0..x8), slightly higher if P is x0. Always on; ignores x9."),
13:("A","Fires when x5 is among the two largest of x0..x8, growing with x5 minus R (third-largest of x0..x8); small baseline from |x0|+..+|x6|."),
14:("A","Codes where the maximum of x0..x8 sits: silent when it is at x4, x6, x7 or x8, active otherwise (strongest at x3, then x5, x1), scaled up with that maximum's value."),
15:("B","Grows as x9 falls below Q, the second-largest of x0..x8, plus a position code: raised when the largest of x0..x8 sits at x1 (most), x3 or x4."),
16:("A","Fires only when x4 is the largest of x0..x8, growing with x4 minus R (third-largest of x0..x8); lowered if the second-largest sits at x2, x3 or x5."),
17:("D","Unclear position code: active unless the largest of x0..x8 sits at x7 or x8 or the second-largest sits at x2, x6 or x8; strongest when the largest is at x2, x4 or x6."),
18:("A","Fires when x9 exceeds Q, the second-largest of x0..x8: about 4.2*x9 - 3.4*Q - 0.55*P (P = largest), but suppressed if that largest sits at x7 or x8."),
19:("A","Fires when x9 is a top-two value overall: about max(0, 4.58*x9 - 1.74*P - 2.75*Q), with P, Q the largest and second-largest of x0..x8."),
20:("B","Grows as x9 falls below the largest of x0..x8 (P), about 1.86:1.39; silenced when P sits at x8, raised when P sits at x2."),
21:("D","Always-on, poorly understood: rises with total magnitude |x0|+..+|x6|, falls with the third-largest of x0..x8 and when x2, x4 or x5 is among the top two."),
22:("A","By how much x9 beats every earlier value: about 6.44*max(0, x9 - max(x0..x8)), i.e. x9 is the overall maximum."),
23:("A","Fires only when x7 is the largest of x0..x8, growing with x7 minus Q (second-largest of x0..x8, i.e. x7's lead) and slightly with |x0|+..+|x7|."),
24:("A","How far x7 lies below Q, the second-largest of x0..x8; zero when x7 is among the top two of x0..x8, reduced when the second-largest sits at x2 or x6."),
25:("B","Fires when x1 or x3 is among the two largest of x0..x8 (strongest when one of them is the largest), scaled up with the largest value of x0..x8."),
26:("A","Lead of the earlier maximum: about 6.68*max(0, max(x0..x7) - max(second-largest of x0..x7, x8)); zero when x8 is the maximum of x0..x8."),
27:("A","Fires when x0 is among the two largest of x0..x8 (weaker when x1 or x2 is), growing with x0; small baseline from |x0|+..+|x6|."),
28:("A","How far x3 lies below Q, the second-largest of x0..x8; zero when x3 is among the top two, raised slightly when the largest sits at x0 or x2 and with |x0|+..+|x7|."),
29:("B","Signals x1 is NOT among the two largest of x0..x8 (grows with Q minus x1), plus a position code raised when the largest of x0..x8 sits at x0, x2 or x3."),
30:("C","Weak, rarely active: on when the largest of x0..x8 sits at x8 AND the second-largest sits at x3, x5, x6 or x7; slightly larger when x9 is small."),
31:("A","Fires when x9 beats Q, the second-largest of x0..x8: about 5.36*max(0, x9 - 0.94*Q), i.e. x9 is among the two largest values."),
}
for k,(c,t) in L.items(): assert len(t.split())<=40,(k,len(t.split()))
src='''import numpy as np
# Notation used by features (all computed from x, an (N,10) array):
#  P8,Q8,R8 = largest, second, third largest of x0..x8;  am, a2 = positions of P8, Q8
#  P7 = max(x0..x7), Q7 = second largest of x0..x7
#  xK = x[:,K];  dK = xK - R8;  qK = xK - Q8
#  AMabc = 1[am in {a,b,c}];  A2abc = 1[a2 in {a,b,c}];  T2abc = 1[am or a2 in {a,b,c}]
#  ABSabc = sum of |x| over positions a,b,c;  G26 = max(0, P7 - max(Q7, x8))
# Each neuron: h = max(0, sum_j b_j * feature_j + b_last)

def _base(x):
    x=np.asarray(x,dtype=float)
    p=x[:,:9]; o=np.argsort(-p,1,kind="stable"); s=np.sort(p,1)[:,::-1]
    return dict(x=x,P8=s[:,0],Q8=s[:,1],R8=s[:,2],am=o[:,0],a2=o[:,1],P7=x[:,:8].max(1),Q7=np.sort(x[:,:8],1)[:,-2])

def _feat(D,name):
    x=D["x"]
    if name in D: return D[name]
    if name[0]=="x": return x[:,int(name[1:])]
    if name[0]=="d": return x[:,int(name[1:])]-D["R8"]
    if name[0]=="q": return x[:,int(name[1:])]-D["Q8"]
    if name.startswith("AM"): return np.isin(D["am"],[int(c) for c in name[2:]])*1.0
    if name.startswith("A2"): return np.isin(D["a2"],[int(c) for c in name[2:]])*1.0
    if name.startswith("T2"):
        S=[int(c) for c in name[2:]]; return (np.isin(D["am"],S)|np.isin(D["a2"],S))*1.0
    if name.startswith("ABS"): return np.abs(x[:,[int(c) for c in name[3:]]]).sum(1)
    if name=="G26": return np.maximum(D["P7"]-np.maximum(D["Q7"],x[:,8]),0)
    raise KeyError(name)

SPEC = %s

LABELS = %s

CONSTANTS = {i: len(SPEC[i][1]) for i in range(32)}

STRUCT = %s

def f(i, x):
    D=_base(x); names,b=SPEC[i]
    z=np.full(D["x"].shape[0],b[-1])
    for n,c in zip(names,b[:-1]): z=z+c*_feat(D,n)
    return np.maximum(z,0)
'''
SPEC={int(k):(v['f'],v['b']) for k,v in sp.items()}
STRUCT={}
for i in range(32):
    fs=SPEC[i][0]
    sets=[n for n in fs if n[:2] in ('AM','A2','T2')]
    msg='features %s chosen by greedy forward selection from a library (values, top-3 of x0..x8, margins, |x| sums, position-set indicators)'%(','.join(fs))
    if sets: msg+='; position sets searched: '+','.join(sets)
    STRUCT[i]=msg
open('labels.py','w').write(src%(repr(SPEC),repr({k:L[k] for k in range(32)}),repr(STRUCT)))
print('written')
