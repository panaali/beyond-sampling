import json
res={r['i']:r for r in json.load(open('resA.json'))+json.load(open('resB.json'))}
res[10]=dict(i=10,form='top2x9',th=[8.41,6.26,0.21],cg=[],ch=[],A=[])
res[2]=dict(i=2,form='win2b',th=[8.19,5.61,-1.86],cg=[],ch=[],A=[])
labels=json.load(open('labeltext.json'))
spec={}
for i in range(16):
    r=res[i]
    spec[i]=dict(form=r['form'],th=[round(t,3) for t in r['th']],cg=r['cg'],ch=r['ch'],A=r['A'])
def nconst(r):
    f=r['form']
    if f=='new':return 1
    if f=='top2x9':return 3
    if f=='win2b':return 3
    return 5
src='''import numpy as np
# Notation: M8, S8 = largest and second-largest of x0..x8; P8, Q8 = their positions (0-based).
# Every formula is ReLU(linear combination of interpretable features).
SPEC = %s
LABELS = %s
CONSTANTS = %s
STRUCT = %s

def _feat(x):
    x = np.asarray(x, dtype=float)
    o = np.argsort(-x[:, :9], 1)
    s = -np.sort(-x[:, :9], 1)
    return o[:, 0], o[:, 1], s[:, 0], s[:, 1], x[:, 9]

def f(i, x):
    x = np.asarray(x, dtype=float)
    P, Q, M, S, x9 = _feat(x)
    sp = SPEC[i]; th = sp["th"]; form = sp["form"]
    if form == "new":
        z = th[0] * (x9 - M)
    elif form == "top2x9":
        z = th[0] * x9 - th[1] * S + th[2] * M
    elif form == "win2b":
        w = -np.sort(-x[:, :5], 1)
        z = th[0] * w[:, 1] + th[1] * np.maximum(x[:, 2], 0) + th[2] * S
    else:
        g = np.array(th[:3])[np.array(sp["cg"])[P]] * M
        if form == "G3S_U":
            z = g + th[3] * S + th[4] * x9
        elif form == "G3H2":
            z = g + np.array(th[3:5])[np.array(sp["ch"])[Q]] * S
        elif form == "G3S_mem":
            z = g + th[3] * S + th[4] * np.maximum(x[:, sp["A"]], 0).sum(1)
    return np.maximum(0.0, z)
'''
L={i:(labels[str(i)][0],labels[str(i)][1]) for i in range(16)}
C={i:nconst(res[i]) for i in range(16)}
ST={i:labels[str(i)][2] for i in range(16)}
open('labels.py','w').write(src%(repr(spec),repr(L),repr(C),repr(ST)))
