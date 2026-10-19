import numpy as np, probe
x=np.random.default_rng(2).standard_normal((100000,10))
h=probe.hidden(x); M=np.maximum.accumulate(x,1)
np.set_printoptions(linewidth=200,suppress=True)
for i in range(16):
    y=h[:,i]; X=np.c_[x,M,np.ones(len(y))]; b=np.linalg.lstsq(X,y,rcond=None)[0]
    print(i,'x:',b[:10].round(1),'M:',b[10:20].round(1),'c',b[20].round(1))
