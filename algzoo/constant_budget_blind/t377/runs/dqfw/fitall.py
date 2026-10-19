import numpy as np, importlib, sys, re
import labels
def r2(y,p): return 1-((y-p)**2).mean()/y.var()
def relufit(y,A,it=10):
    act=y>0
    for _ in range(it):
        c,*_=np.linalg.lstsq(A[act],y[act],rcond=None); act=(A@c>0)|(y>0)
    return c
x=np.load('x1.npy'); h=np.load('h1.npy')
d=labels._base(x)
only=[int(a) for a in sys.argv[1:]] or list(range(32))
C=dict(labels.C)
for i in only:
    mode,fn=labels.SPEC[i]; A=np.stack(fn(d),1)
    if mode=='lin': c,*_=np.linalg.lstsq(A,h[:,i],rcond=None)
    else: c=relufit(h[:,i],A)
    C[i]=[float('%.4g'%v) for v in c]
src=open('labels.py').read()
block='C = {\n'+''.join('    %d: %s,\n'%(k,C[k]) for k in sorted(C))+'}\n'
src=re.sub(r'# BEGIN CONSTS\n.*?# END CONSTS',lambda m:'# BEGIN CONSTS\n'+block+'# END CONSTS',src,flags=re.S)
open('labels.py','w').write(src)
importlib.reload(labels)
import probe
xt=np.random.default_rng(5).standard_normal((50000,10)); ht=probe.hidden(xt)
np.save('xt.npy',xt); np.save('ht.npy',ht)
for i in only:
    print(i,'R2 test %.3f'%r2(ht[:,i],labels.f(i,xt)),C[i])
