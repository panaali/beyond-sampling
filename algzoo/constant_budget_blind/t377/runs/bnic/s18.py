import numpy as np
import labels_fit as L
x=np.load('x1.npy')[:50000]; h=np.load('h1.npy')[:50000]
M=np.maximum.accumulate(x,1); m=M[:,9]; am=np.argmax(x,1); T=m[:,None]-x; I=np.eye(10)[am]
G9=np.maximum(x[:,9]-M[:,8],0)
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
for i in range(16):
    y=h[:,i]; B=np.stack(L._terms(i,x),1)
    r0,_=fit(B,y); r1,_=fit(np.c_[B,m[:,None]*I],y); r2,c2=fit(np.c_[T,G9,m[:,None]*I],y)
    print(i,'cur %.3f +amI %.3f | T+G9+amI %.3f'%(r0,r1,r2), 'Tw',c2[:10].round(1),'G9',c2[10].round(1),'amw',c2[11:].round(1))
