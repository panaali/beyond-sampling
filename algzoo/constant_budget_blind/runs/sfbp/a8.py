import numpy as np,sys,pickle
from plib import plib
d=np.load('d4.npz'); x=d['x'][:10000]
F=plib(x); names=list(F); A=np.stack([F[n] for n in names],1)
W={0:-1.549,6:-1.89,7:8.8,9:-.338,12:-3.065,13:-1.699,16:5.707,18:-1.46,21:4.402,23:4.875,29:-1.667,2:-.151,3:-.315,4:-.202,10:-.288}
def getc(i):
    w=W.get(i,0); n=len(x)
    if w>0.5: return d['Hp'][:n,i]-40*w, np.abs((d['Hp2'][:n,i]-d['Hp'][:n,i])/10-w)<1e-3, False
    if w<-0.1: return d['Hn'][:n,i]+40*w, np.abs((d['Hn2'][:n,i]-d['Hn'][:n,i])/-10-w)<1e-3, False
    return d['H0'][:n,i], None, True
def score(X,y,cens,m):
    if not cens:
        co=np.linalg.lstsq(X[m],y[m],rcond=None)[0]; return 1-((X[m]@co-y[m])**2).mean()/y[m].var(),co
    mm=y>1e-6
    for it in range(6):
        co=np.linalg.lstsq(X[mm],y[mm],rcond=None)[0]; mm=(y>1e-6)|(X@co>0)
    return 1-((np.maximum(X@co,0)-y)**2).mean()/y.var(),co
out={}
for i in map(int,sys.argv[1:]):
    y,m,cens=getc(i); ch=[]; log=[]
    for st in range(5 if cens else 4):
        best=None
        for k in range(A.shape[1]):
            if k in ch: continue
            R,co=score(A[:,ch+[k]],y,cens,m)
            if best is None or R>best[0]: best=(R,k,co)
        ch.append(best[1]); log.append(f'{names[best[1]]}:{best[0]:.3f}')
    out[i]=([names[k] for k in ch],best[2])
    print(i,'cens' if cens else 'exact',' '.join(log),np.round(best[2],2),flush=True)
pickle.dump(out,open('greedy_%s.pkl'%sys.argv[1],'wb'))
