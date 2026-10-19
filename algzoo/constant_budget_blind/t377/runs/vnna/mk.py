import json
F=json.load(open('final.json'))
MS="M = largest of x0..x8, S = second-largest of x0..x8"
L={
0:("B","How far x9 falls below the earlier values: ReLU(0.40M+0.87S-1.53x9), with M weighted more if M sits at x1,x2,x5,x6; zero if M is x8. "+MS+"."),
1:("A","Gap between the top two of x0..x8: about 4.6*(max of x0..x7) - 4.3*S, so it is large when one early value clearly dominates and near zero when x8 is the largest; ignores x9. S = second-largest of x0..x8."),
2:("A","Detector for x3 being large: grows with x3, boosted when x3 is the largest or second-largest of all ten inputs, and reduced slightly by the sum of the other inputs."),
3:("B","Signals an early maximum: grows with M when M is at x0..x6 (more for x0..x4), reduced by S when S is at x0..x4, and by x9; zero if S is x8. M, S = largest, second-largest of x0..x8."),
4:("D","Not well understood. Mainly tracks S when S sits at an even position (x0,x2,x4,x6, partly x8), plus a little M, minus a little x9; possibly a parity-like code of rank positions. S = second-largest of x0..x8, M = largest."),
5:("A","Margin of the largest over the second-largest among x0..x8: about 4.1M - 2.4S, smaller when M is at x2,x5,x6 or x7, slightly reduced by the sum of x0..x8; ignores x9. "+MS+"."),
6:("B","How far x9 falls below the earlier values: ReLU(0.57M+0.92S-1.87x9), with M weighted more when M is at x0,x2,x5 or x7. Zero when x9 is the overall maximum. "+MS+"."),
7:("A","Fires only when x9 is the new overall maximum: about 8.8 times how much x9 exceeds the largest of x0..x8; zero otherwise."),
8:("B","Weighted level of the largest of x0..x8 (M), stronger when M is early (x0,x1,x2) or at x5, plus the second-largest when it sits at x0,x1,x5, minus about 0.3 times the sum of x0..x8; ignores x9."),
9:("A","Almost always silent. Fires only when the second-largest of x0..x8 (and the third-largest) are negative, i.e. when most early inputs are low; slightly reduced by x9."),
10:("B","Active only when the largest of x0..x8 (M) is at x0,x4 or x5: grows with M when M is at x4/x5, with the second-largest when it is at x0..x2, shrinks when that second-largest is at x5..x8 and with x9."),
11:("A","Position-of-maximum code: proportional to the largest of x0..x8 (M) when M is at x0,x4 or x6 (strongest at x4, weaker at x1,x2), minus a little of the second-largest; zero when M is at x3,x5,x7,x8."),
12:("A","Fires when x9 is not in the top two: about 3.1 times how far x9 lies below the second-largest of x0..x8; zero if x9 exceeds that second-largest."),
13:("B","How far x9 falls below the earlier values: ReLU(0.53M+0.91S-1.70x9), with M weighted more if M sits at x1,x2,x4,x7; zero if M is x8. "+MS+"."),
14:("A","Detector for x8 entering the top two of x0..x8: grows with x8 and with the gap below the earlier maximum, steeply suppressed when x8 is below the second-largest of x0..x7, minus a little of the sum of x0..x7."),
15:("A","Detector for x1 being large: grows with x1, boosted when x1 is the largest or second-largest of all ten inputs, and reduced slightly by the sum of the other inputs."),
16:("A","Fires when x9 enters the top two: about 5.7 times how much x9 exceeds the second-largest of x0..x8; zero otherwise."),
17:("A","Running-maximum register: about 8.1 times the largest of x0..x8 plus 2.0 times the second-largest of x0..x8; always active, ignores x9."),
18:("B","How far x9 falls below the earlier values: ReLU(0.43M+0.66S-1.43x9), with M weighted much more when M is at x1 (and somewhat at x0). Zero when x9 is the overall maximum. "+MS+"."),
19:("A","Detector for x6 being large: grows with x6, boosted when x6 is the largest or second-largest of all ten inputs, and reduced slightly by the sum of the other inputs."),
20:("A","Level of the top two of x0..x8 when x0 is not among them (zero if x0 is largest or second-largest of x0..x8); reduced by x0 and by the sum of x0..x8, boosted when the largest is x1."),
21:("A","x9 strength relative to the top two earlier values: ReLU(4.2x9 - 2.2M - 2.0S), active mostly when x9 is the overall maximum by a clear margin. "+MS+"."),
22:("A","Detector for x7 being large, mainly when x7 is the overall maximum (or second-largest): grows with x7 in those cases, reduced slightly by the sum of the other inputs."),
23:("A","x9 entering the top two: ReLU(4.7x9 - 1.0M - 3.7S), zero when the largest of x0..x8 is x8. "+MS+"."),
24:("A","Level of the top two of x0..x7 when x5 is not among them (zero if x5 is largest or second-largest of x0..x7); reduced by x5 and slightly by the sum of x0..x7; ignores x8, x9."),
25:("A","Level of the second-largest of x0..x8, weighted more when it occurs early, plus a little of the largest, when x8 is not among the top two of x0..x8 (zero otherwise); ignores x9."),
26:("B","Detector for x2 being large, combined with the largest of x0..x8 (M) weighted by its position: strong when M is x2, moderate when M is x0 or x4; minus a fraction of the sum of x0..x8."),
27:("A","Position-of-maximum code: active only when the overall maximum is x2 or x6, roughly 1.5 times the maximum minus 0.55 times the second-largest; slightly reduced by x7."),
28:("B","Level of the top values of x0..x8, larger when the largest sits at x0,x1 or x3, zero when the largest is x8, minus about 0.11 times the sum of x0..x8; ignores x9."),
29:("B","How far x9 falls below the earlier values: ReLU(0.35M+1.03S-1.66x9), with M weighted more if M sits at x0,x3,x6,x7; zero if M is x8. "+MS+"."),
30:("A","Level of the top two of x0..x8 when x4 is not among them (zero if x4 is largest or second-largest of x0..x8); reduced by x4 and the sum of x0..x8, boosted when the largest is x0 or x3."),
31:("A","Second-maximum register: about 5.5 times the second-largest of x0..x8, plus a smaller share of the largest (more when it is at x0,x1,x3,x4), minus a little of the sum; ignores x9."),
}
ST={0:"position set {1,2,5,6} and gate (argmax of x0..x8 != 8) chosen from per-position fits",
1:"choice of max(x0..x7) and second(x0..x8)",2:"feature choice (indicators am==3,a2==3, sum of others)",
3:"sets {0..4} for max and second positions; gates max-position<7, second-position!=8",
4:"sets {0,2,4,6,8},{0,2,4,6} for second position, {0,1,2,3,4,8} for max position",
5:"set {2,5,6,7} of max positions",6:"set {0,2,5,7} of max positions",7:"",8:"sets {0,1,2,5},{5} for max position, {0,1,5} for second position",
9:"feature choice (2nd and 3rd largest of x0..x8, x9)",10:"gate max position in {0,4,5}; sets {4,5}, {5,6,7,8}, {0,1,2}",
11:"gate max position in {0,1,2,4,6}; sets {0,4,6},{4},{1,2}",12:"",13:"set {1,2,4,7}; gate max-position != 8",
14:"comparison features ReLU(prefix max - x8), ReLU(prefix second - x8) with prefix x0..x7",15:"as neuron-2 template for position 1",
16:"",17:"",18:"sets {1},{0} of max positions",19:"as detector template for position 6",20:"gate x0 not top-2 of x0..x8; set {1}",
21:"",22:"as detector template for position 7",23:"gate max-position of x0..x8 != 8",24:"prefix x0..x7 (since x8,x9 have ~0 gradient); gate x5 not top-2 of x0..x7",
25:"gate x8 not top-2 of x0..x8; linear-in-position weight for S",26:"sets {0,2,4},{2} for max position",27:"set {2,6} of overall argmax",
28:"set {0,1,3}; gate max-position != 8",29:"set {0,3,6,7}; gate max-position != 8",30:"gate x4 not top-2 of x0..x8; set {0,3}",31:"set {0,1,3,4}"}
src=['"""Neuron labels and formulas for the 32-unit second-largest-position RNN (final hidden state).',
'Helper notation inside formulas: M9/S9 = largest/second-largest of x0..x8, am9/a29 their positions,',
'T1/T2 = largest/second-largest of all ten, am/a2 their positions, P(L,j) = j-th largest of x0..x_{L-1},',
'PA(L,j) its position, SM(L) = x0+..+x_{L-1}, pm(k) = max of x0..x_{k-1}, mo(k)/so(k) = max/second of all but x_k."""',
'import numpy as np','',
'def R(z):','    return np.maximum(z, 0.0)','',
'def _ns(x):',
'    x = np.asarray(x, dtype=float)',
'    s = np.argsort(-x, 1); srt = -np.sort(-x, 1)',
'    p = x[:, :9]; s9 = np.argsort(-p, 1); sr9 = -np.sort(-p, 1)',
'    d = dict(x=x, am=s[:, 0], a2=s[:, 1], T1=srt[:, 0], T2=srt[:, 1], T3=srt[:, 2],',
'             am9=s9[:, 0], a29=s9[:, 1], M9=sr9[:, 0], S9=sr9[:, 1], x9=x[:, 9])',
'    d["P"] = lambda L, j: -np.sort(-x[:, :L], 1)[:, j - 1]',
'    d["PA"] = lambda L, j: np.argsort(-x[:, :L], 1)[:, j - 1]',
'    d["SM"] = lambda L: x[:, :L].sum(1)',
'    d["pm"] = lambda k: x[:, :k].max(1)',
'    d["mo"] = lambda k: np.delete(x, k, 1).max(1)',
'    d["so"] = lambda k: np.sort(np.delete(x, k, 1), 1)[:, -2]',
'    d["isin"] = lambda a, S: np.isin(a, list(S)).astype(float)',
'    return d','']
src.append('LABELS = {'); [src.append('    %d: (%r, %r),'%(i,L[i][0],L[i][1])) for i in range(32)]; src.append('}')
src.append('CONSTANTS = {'); [src.append('    %d: %d,'%(i,F[str(i)]['n'])) for i in range(32)]; src.append('}')
src.append('STRUCT = {'); [src.append('    %d: %r,'%(i,ST[i])) for i in range(32)]; src.append('}')
src.append('FITTED = {'); [src.append('    %d: %r,'%(i,[round(v,4) for v in F[str(i)]['p']])) for i in range(32)]; src.append('}')
src.append('EXPR = {'); [src.append('    %d: %r,'%(i,F[str(i)]['e'])) for i in range(32)]; src.append('}')
src+=['','def f(i, x):','    d = _ns(x)','    d["p"] = FITTED[i]','    out = eval(EXPR[i], {"np": np, "R": R}, d)','    return np.asarray(out, dtype=float) * np.ones(len(d["x"]))','']
open('labels.py','w').write("\n".join(src))
print([(i,len(L[i][1].split())) for i in range(32) if len(L[i][1].split())>40])
