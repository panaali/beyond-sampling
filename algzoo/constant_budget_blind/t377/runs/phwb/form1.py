import numpy as np, sys
from lib import pf,relufit
x=np.load('x0.npy')[:20000];h=np.load('h0.npy')[:20000];F=pf(x);one=np.ones(len(x))
M8,s8,p8,x9=F['M8'],F['s8'],F['p8'],F['x9'];D=M8-s8
forms={'a':lambda T:[M8,D*T,x9,one],'b':lambda T:[M8,D*T,D,x9,one],'c':lambda T:[M8,D*T,s8*T,x9,one]}
for i in [int(a) for a in sys.argv[1].split(',')]:
    y=h[:,i]; g=y>0.3
    for fn,fm in forms.items():
        best=(-9,)
        for w in np.arange(0.3,1.6,0.05):
            for f1 in np.arange(-2,10.1,0.25):
                A=np.c_[tuple(fm(np.cos(w*(p8-f1))))]
                c,_,_,_=np.linalg.lstsq(A[g],y[g],rcond=None)
                r=1-((A[g]@c-y[g])**2).mean()/y[g].var()
                if r>best[0]: best=(r,w,f1)
        r,w,f1=best; A=np.c_[tuple(fm(np.cos(w*(p8-f1))))]; c,rr=relufit(A,y)
        print(i,fn,f"w={w:.2f} f={f1}",np.round(c,2),'R2',round(rr,4),flush=True)
