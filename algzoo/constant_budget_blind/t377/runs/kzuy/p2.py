import numpy as np, probe
x=np.random.default_rng(2).standard_normal((100000,10))
h=probe.hidden(x); s=np.sort(x,1)[:,::-1]; am=np.argmax(x,1)
def r2(X,y):
    X=np.c_[X,np.ones(len(y))]; b=np.linalg.lstsq(X,y,rcond=None)[0]; return 1-((y-X@b)**2).mean()/y.var(),b
# running maxima: M_t = max(x0..x_t)
M=np.maximum.accumulate(x,1)
for i in range(16):
    y=h[:,i]
    a=r2(x,y)[0]; b=r2(np.c_[x,s],y)[0]; c=r2(np.c_[x,M],y)[0]; d=r2(np.c_[x,M,s],y)[0]
    # per argmax group linear in x
    e=0
    for t in range(10):
        m=am==t; e+=((y[m]-np.c_[x[m],np.ones(m.sum())]@np.linalg.lstsq(np.c_[x[m],np.ones(m.sum())],y[m],rcond=None)[0])**2).sum()
    e=1-e/len(y)/y.var()
    print(i,f'lin {a:.3f} +sorted {b:.3f} +runmax {c:.3f} +both {d:.3f} perAm {e:.3f}')
