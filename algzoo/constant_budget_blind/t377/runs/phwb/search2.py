import numpy as np, sys
from lib import pf,relufit
x=np.load('x0.npy')[:20000];h=np.load('h0.npy')[:20000];F=pf(x);one=np.ones(len(x))
M8,s8,p8,q8=F['M8'],F['s8'],F['p8'],F['q8']
for i in [int(a) for a in sys.argv[1].split(',')]:
    y=h[:,i]; g=y>0.3; best=(-9,)
    for w in np.arange(0.3,1.31,0.1):
        for f1 in np.arange(0,9.1,0.5):
            T1=np.cos(w*(p8-f1))
            for f2 in np.arange(0,9.1,1.0):
                T2=np.cos(w*(q8-f2))
                A=np.c_[M8*T1,s8*T2,M8,s8,one]
                c,_,_,_=np.linalg.lstsq(A[g],y[g],rcond=None)
                r=1-((A[g]@c-y[g])**2).mean()/y[g].var()
                if r>best[0]: best=(r,w,f1,f2)
    r,w,f1,f2=best
    A=np.c_[M8*np.cos(w*(p8-f1)),s8*np.cos(w*(q8-f2)),M8,s8,one]
    c,rr=relufit(A,y)
    print(i,f"w={w:.1f} f1={f1} f2={f2}",np.round(c,2),'R2',round(rr,3),flush=True)
