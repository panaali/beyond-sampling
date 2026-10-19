import numpy as np, json
from featcore import _feat
from lib import feats
d={**json.load(open('fit_0.json')),**json.load(open('fit_16.json'))}
x=np.random.default_rng(5).standard_normal((3000,10)); F=feats(x)
for i in range(32):
    for c in d[str(i)]['cols']:
        e=np.abs(_feat(c,x)-F[c]).max()
        if e>1e-9: print('MISMATCH',i,c,e)
print('done')
