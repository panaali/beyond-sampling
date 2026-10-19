import numpy as np, json
from featlib import *
np.set_printoptions(precision=2,suppress=True,linewidth=200)
x=np.load('x1.npy');H=np.load('H1.npy')
fit=json.load(open('fitted.json'))
a=am9(x);b=a2_9(x)
rk10=np.argsort(np.argsort(-x,1),1)
for i,j in [(4,4),(0,2),(13,5),(27,0)]:
    kind,feats,gate,c=fit[str(i)]
    F=np.column_stack([eval(e) for e in feats]); p=relu(F@np.array(c[:-1])+c[-1]); r=H[:,i]-p
    for nm,m in [('top1',a==j),('top2',b==j),('rest',(a!=j)&(b!=j))]:
        print(i,nm,'resid by a2' if nm=='top1' else 'resid by am', [round(r[m&((b if nm=='top1' else a)==k)].mean(),2) for k in range(9)], 'by rank of x9', [round(r[m&(rk10[:,9]==k)].mean(),2) for k in range(4)])
        # number of later elements greater than x_j - 1
