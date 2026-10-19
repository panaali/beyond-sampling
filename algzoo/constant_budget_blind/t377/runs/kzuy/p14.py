import numpy as np, probe
from dic5 import dic5
from dic3 import fitrelu
F={3:['x1','M7','r(x2)','r(-x4)','Mpe6'],9:['x4','x6','M7','r(x2)','xn5'],12:['r(x3)','M1-x1','xn3','r(x2-M1)','x3'],
   13:['M8','r(x8-M7)','x2','x4','r(x1-M0)'],15:['M5-x5','r(x2)','r(x6-M5)','r(x1-M0)','PE*g'],4:['M7','Mpe6','xn6','r(x1-M0)','po*(M9-x4)']}
x=np.random.default_rng(12).standard_normal((30000,10)); h=probe.hidden(x); D=dic5(x); am=np.argmax(x,1)
for i,ks in F.items():
    X=np.stack([D[k] for k in ks],1); b=fitrelu(X,h[:,i]); res=h[:,i]-np.maximum(X@b,0)
    print(i,'R2 %.3f'%(1-res.var()/h[:,i].var()),'res var by am:',[round((res[am==t]**2).mean()/h[:,i].var(),3) for t in range(10)])
    c=sorted(((abs(np.corrcoef(v,res)[0,1]),k) for k,v in D.items() if v.std()>0),reverse=True)[:6]
    print('   resid corr:',[(k,round(v,2)) for v,k in c])
