import json
from labtext import L
F=json.load(open('final.json'))
fl=open('featlib.py').read()
for i,(c,t) in L.items(): assert len(t.split())<=40,(i,len(t.split()))
s='"""Neuron labels and formulas for h_10 (16 neurons). Only numpy is used."""\n'+fl+'\n\n'
s+='LABELS = {\n'+''.join(f'    {i}: ({L[i][0]!r}, {L[i][1]!r}),\n' for i in range(16))+'}\n\n'
s+='FORMULAS = {\n'+''.join(f'    {i}: ({F[str(i)][0]!r}, {F[str(i)][1]!r}),\n' for i in range(16))+'}\n\n'
s+='CONSTANTS = {i: len(FORMULAS[i][1]) for i in range(16)}\n\n'
s+=('STRUCT = {i: "feature set chosen by greedy/beam search over a dictionary of max/rank/position features "\n'
    '             "(running maxima M_k, capped values, pairwise max/min, ReLU differences, argmax-parity indicators); "\n'
    '             "features: " + ", ".join(FORMULAS[i][0]) + "; output passed through ReLU" for i in range(16)}\n\n')
s+=('def f(i, x):\n    """Predict h_10[i]: ReLU of a weighted sum of named features (see feature())."""\n'
    '    x = np.asarray(x, float)\n    names, w = FORMULAS[i]\n'
    '    z = sum(wk * feature(n, x) for n, wk in zip(names, w))\n    return np.maximum(z, 0.0)\n')
open('labels.py','w').write(s)
