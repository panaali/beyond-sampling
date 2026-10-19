import json
fit=json.load(open('fitted.json'))
T="top two of x0..x8"
L={
0:("B","Fires when x2 is among the two largest of x0..x8 (strongest if it is the largest), growing with x2; also a smaller boost when x0 is the largest of x0..x8."),
1:("B","Positive when x9 lies below a level set by the two largest of x0..x8 (about 1.4*second + 0.4*max - 1.9*x9); level raised when the largest of x0..x8 is x0, x2 or x3."),
2:("B","Positive when x9 lies below a mix of the largest and second-largest of x0..x8 (about 1.2*max + 0.9*second - 2*x9); switched off when x8 is the largest of x0..x8."),
3:("A","Silent if x8 is among the two largest of x0..x8; otherwise grows as x8 falls below the second-largest of x0..x8 (about 1.33*second - 1.45*x8), a bit higher if the largest is x5 or x6."),
4:("B","Fires when x4 is among the two largest of x0..x8 (more if it is the largest), growing with x4; a smaller boost when the largest of x0..x8 is x1 or x2."),
5:("A","Always on: weighted sum of the largest and second-largest of x0..x8 (about 3.3*max + 2.1*second), plus a small bump when the largest of x0..x8 is x1, x5 or x7."),
6:("A","How far x9 falls below the second-largest of x0..x8: about 3*(second - x9) (+0.2*max), rectified; zero when x9 is among the two largest of all ten inputs."),
7:("A","Fires when x6 is among the two largest of x0..x8 (more if it is the largest), growing with x6 and decreasing with the sum of the other inputs x0..x8."),
8:("B","Positive when x9 lies below a level set by the top two of x0..x8 (about 0.8*max + 1.1*second - 1.9*x9); level raised when the largest of x0..x8 is at an even position (x0,x2,x4,x6,x8)."),
9:("A","Amount by which x8 exceeds all earlier inputs: about 7.25*ReLU(x8 - max(x0..x7)). Does not depend on x9."),
10:("A","Silent when x8 is among the two largest of x0..x8; otherwise a moderate level that rises with the second-largest of x0..x7 and drops the later the largest of x0..x8 occurs."),
11:("D","Poorly understood, low-variance: rises when x8 is the second-largest of x0..x8 (exceeds second-largest of others but not the max) and when x2 is the largest; reduced when x8 is the largest."),
12:("A","Always on: weighted sum of the largest and second-largest of x0..x8 (about 5.7*max + 3.5*second), plus a small bump when the largest of x0..x8 is x0, x2, x3, x4 or x6."),
13:("A","Fires when x5 is among the two largest of x0..x8, growing with x5 and with how far x5 exceeds the second-largest of the other values in x0..x8."),
14:("B","Active only when the largest of x0..x8 sits at x0, x1, x2, x3 or x5; then about 0.8*max - 0.37*second of x0..x8 + 0.3, with an extra boost when x3 is the largest."),
15:("B","Positive when x9 lies below a level set by the top two of x0..x8 (about 0.5*max + 1.4*second - 2*x9); level raised when the largest of x0..x8 is x1 or x4."),
16:("A","Fires only when x4 is the largest of x0..x8; then about 0.9*x4 - 0.11*(sum of the other inputs x0..x8) + 0.15."),
17:("D","Poorly understood position mix: on when x6 is the overall largest or x7 is second-largest of x0..x8; silenced when x8 is among the top two or x7/x8 is the largest."),
18:("A","x9 rising above the second-largest of x0..x8 (about 4*x9 - 3.2*second - 0.6*max), rectified; only when the largest of x0..x8 is in x0..x6 (off if it is x7 or x8)."),
19:("A","x9 rising above a mix of the top two of x0..x8 (about 4.5*x9 - 2.9*second - 1.6*max), rectified; slightly higher when the largest of x0..x8 is x2, x3, x5, x6 or x8."),
20:("B","Positive when x9 lies below a level set by the top two of x0..x8 (about 0.85*max + 1.0*second - 1.8*x9); switched off when x8 is the largest of x0..x8."),
21:("D","Poorly understood, always on: roughly decreases with the sum of x0..x7 and increases with the three largest values of x0..x8."),
22:("A","Amount by which x9 exceeds all earlier inputs: about 6.5*ReLU(x9 - max(x0..x8)), i.e. x9 is the overall maximum."),
23:("A","Fires only when x7 is the largest of x0..x8; then about 1.07*x7 - 0.14*(sum of the other inputs x0..x8)."),
24:("A","Silent if x7 is among the two largest of x0..x8; otherwise grows as x7 falls below the second-largest of the other values in x0..x8 (about 2.1*second - 1.2*x7)."),
25:("B","Fires when x1 or x3 is among the two largest of x0..x8, growing with how far each exceeds the second-largest of the other values; strongest when x1 is the largest."),
26:("A","Gap between the largest and second-largest of x0..x8, about 6.7*(max - second), rectified; zero when x8 itself is the largest of x0..x8."),
27:("A","Fires when x0 is among the two largest of x0..x8 (more if it is the largest), growing with x0 and decreasing with the sum of the other inputs x0..x8."),
28:("A","Silent if x3 is among the two largest of x0..x8; otherwise grows with the top two values of x0..x8 and as x3 decreases; lower when the largest of x0..x8 is x1 or x5..x8."),
29:("B","High when x1 is NOT among the two largest of x0..x8: drops steeply as x1 exceeds the second-largest of the others, rises with max(x0..x3), and with x0 exceeding the second-largest of others."),
30:("A","Fires only when the two largest of all ten inputs are x8 and one of x3, x5, x6, x7; larger when x8 is the largest and with a bigger second-to-third gap; reduced by large x9."),
31:("A","Amount by which x9 exceeds the second-largest of x0..x8: about ReLU(5.36*x9 - 5.05*second - 0.13); positive means x9 is among the two largest of all ten inputs."),
}
ST={}
for k,(kind,feats,gate,c) in fit.items():
    s='form=%s; features=%s'%(kind,feats)+('; gate=%s'%gate if gate else '')
    ST[int(k)]=s
for i in L: assert len(L[i][1].split())<=40,(i,len(L[i][1].split()))
lib=open('featlib.py').read()
out=['"""Per-neuron labels and formulas for h_10 of the second-largest-position RNN (numpy only)."""',lib,'',
'SPEC = {']
for k,(kind,feats,gate,c) in fit.items():
    out.append('    %s: (%r, %r, %r, %r),'%(k,kind,feats,gate,[round(v,4) for v in c]))
out.append('}')
out.append('LABELS = {')
for i in range(32): out.append('    %d: (%r, %r),'%(i,L[i][0],L[i][1]))
out.append('}')
out.append('CONSTANTS = {i: len(SPEC[i][3]) for i in range(32)}')
out.append('STRUCT = {')
for i in range(32): out.append('    %d: %r,'%(i,ST[i]))
out.append('}')
out.append('''
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
''')
open('labels.py','w').write('\n'.join(out))
