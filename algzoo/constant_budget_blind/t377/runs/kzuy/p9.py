import numpy as np, probe
np.set_printoptions(linewidth=200,suppress=True)
x=np.random.default_rng(9).standard_normal((200000,10)); h=probe.hidden(x)[:,14]
am=np.argmax(x,1); m=am==6; x=x[m]; y=h[m]; s=np.sort(x,1)[:,::-1]
M5=x[:,:6].max(1)
print('n',m.sum(),'frac+',(y>0).mean())
for name,F in [('x6-M5',x[:,6]-M5),('x6',x[:,6]),('x6-s1',x[:,6]-s[:,1])]:
    print(name,np.corrcoef(F,y)[0,1])
X=np.c_[x[:,6],M5,x[:,7],x[:,8],x[:,9]]
b=np.linalg.lstsq(X,y,rcond=None)[0]; print('x6,M5,x7,x8,x9',b.round(2),1-((y-X@b)**2).mean()/y.var())
# binned by x6-M5 and x9
d=x[:,6]-M5
for lo in [0,0.5,1,2]:
    mm=(d>lo)&(d<lo+0.5); print('d bin',lo,y[mm].mean().round(2), np.corrcoef(x[mm,9],y[mm])[0,1].round(2))
