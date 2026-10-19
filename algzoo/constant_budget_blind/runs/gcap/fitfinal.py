import numpy as np, sys, importlib
sys.path.insert(0,'.')
from lib import relufit, r2
x=np.load('x.npy');H=np.load('H.npy')
FEATS={0:['LEVEL','x3','x4','x5','x8'],1:['G6','LEVEL','D46','g6','D48'],2:['LEVEL','x5','x3','g6','D89'],
3:['G5','x7','D35','D47','D17'],4:['gap8','gap9'],5:['gap9'],6:['LEVEL','x2','x4','x8','D24'],7:['LEVEL','x5','x6','x9','g4']}
tpl=open('labels_template.py').read()
exec(tpl.replace('__SPEC__','{}'))
spec={}
for i,names in FEATS.items():
    F=np.c_[[feat(x,n) for n in names]].T
    R,c,p=relufit(F,H[:,i],it=15)
    spec[i]=(names,[round(float(v),4) for v in c]); print(i,round(R,4),spec[i])
open('labels_core.py','w').write(tpl.replace('__SPEC__',repr(spec)))
