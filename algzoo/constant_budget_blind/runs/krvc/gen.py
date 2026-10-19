import json
from labtext import L
d={**json.load(open('fit_0.json')),**json.load(open('fit_16.json'))}
core=open('featcore.py').read()
SPEC={i:(d[str(i)]['cols'],[round(v,5) for v in d[str(i)]['c']]) for i in range(32)}
out=['"""Neuron labels and formulas for h_10 of the second-largest-position RNN.',
'Each formula is relu(sum_k c_k * feature_k + c0) over interpretable input features (see _feat).','"""',core,'',
'SPEC = '+repr(SPEC),'',
'LABELS = '+repr({i:L[i] for i in range(32)}),'',
'CONSTANTS = '+repr({i:len(SPEC[i][1]) for i in range(32)}),'',
'STRUCT = '+repr({i:('features chosen by greedy search over a library of order-statistic/position features: '+', '.join(SPEC[i][0])) for i in range(32)}),'',
'''def f(i, x):
    x = np.asarray(x, dtype=float)
    cols, c = SPEC[i]
    z = np.full(len(x), c[-1])
    for name, w in zip(cols, c[:-1]):
        z = z + w * _feat(name, x)
    return np.maximum(z, 0.0)
''']
open('labels.py','w').write('\n'.join(out))
for i in range(32): assert len(L[i][1].split())<=40,(i,len(L[i][1].split()))
