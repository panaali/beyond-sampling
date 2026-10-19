import numpy as np, sys, time
from lib import pf,relufit
x=np.load('x0.npy')[:15000];h=np.load('h0.npy')[:15000];F=pf(x)
D=F['M8']-F['s8']; I8=(F['p8']==8).astype(float); one=np.ones(len(x))
def cols(w,ph):
    T=np.cos(w*(F['p8']-ph))
    return {'1':one,'M8':F['M8'],'s8':F['s8'],'x9':F['x9'],'x8':F['x8'],'t8':F['t8'],'D':D,
            'DT':D*T,'M8T':F['M8']*T,'s8T':F['s8']*T,'T':T,'DI8':D*I8,'M8I8':F['M8']*I8,'I8':I8,'M7':F['M7']}
def r2ls(A,y,g):
    c,_,_,_=np.linalg.lstsq(A[g],y[g],rcond=None); return 1-((A[g]@c-y[g])**2).mean()/y[g].var()
neurons=[int(a) for a in sys.argv[1].split(',')]; K=int(sys.argv[2]) if len(sys.argv)>2 else 5
for i in neurons:
    y=h[:,i]; g=y>0.3; best=(-9,)
    for w in np.arange(0.3,1.31,0.1):
        for ph in np.arange(-1,9.01,0.5):
            C=cols(w,ph); names=list(C); sel=[]
            for k in range(K):
                sc=[(r2ls(np.c_[tuple(C[n] for n in sel+[m])],y,g),m) for m in names if m not in sel]
                s,m=max(sc); sel.append(m)
            if s>best[0]: best=(s,w,ph,sel)
    s,w,ph,sel=best; C=cols(w,ph); A=np.c_[tuple(C[n] for n in sel)]
    c,r=relufit(A,y)
    print(i,f"w={w:.1f} ph={ph:.1f}",sel,np.round(c,3),"R2relu",round(r,3),flush=True)
