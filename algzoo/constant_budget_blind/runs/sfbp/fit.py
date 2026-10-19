import numpy as np, sys
from plib import plib
d=np.load('d4.npz'); x=d['x']; F=plib(x)
W={0:-1.549,6:-1.89,7:8.8,9:-.338,12:-3.065,13:-1.699,16:5.707,18:-1.46,21:4.402,23:4.875,29:-1.667,2:-.151,3:-.315,4:-.202,10:-.288}
def C(i):
    w=W.get(i,0)
    if w>0.5: return d['Hp'][:,i]-40*w, np.abs((d['Hp2'][:,i]-d['Hp'][:,i])/10-w)<1e-3
    if w<-0.1: return d['Hn'][:,i]+40*w, np.abs((d['Hn2'][:,i]-d['Hn'][:,i])/-10-w)<1e-3
    return d['H0'][:,i], d['H0'][:,i]>1e-6
def fitc(i,names,mask=None,intercept=False):
    c,w=C(i); A=np.stack([F[n] for n in names],1)
    if intercept: A=np.c_[A,np.ones(len(c))]
    m=w if mask is None else mask
    co=np.linalg.lstsq(A[m],c[m],rcond=None)[0]; r2=1-((A[m]@co-c[m])**2).mean()/c[m].var()
    print(i,names,co.round(3),'R2=%.4f'%r2,'n=%d'%m.sum())
if __name__=='__main__':
    for arg in sys.argv[1:]:
        i,ns=arg.split(':'); fitc(int(i),ns.split(','))
