import numpy as np
x=np.load('x1.npy'); h=np.load('h1.npy')
N=len(x)
M=np.maximum.accumulate(x,1)
srt=-np.sort(-x,1)
def r2(F,y):
    F=np.c_[F,np.ones(len(y))]; c,*_=np.linalg.lstsq(F,y,rcond=None); e=y-F@c; return 1-e.var()/y.var(),c
libs={'x':x,'M':M,'sorted':srt,'x+M':np.c_[x,M],'x+M+srt':np.c_[x,M,srt]}
for i in range(16):
    print(i,' '.join(f"{k}:{r2(F,h[:,i])[0]:.3f}" for k,F in libs.items()))
