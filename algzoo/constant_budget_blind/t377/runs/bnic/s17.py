import numpy as np, itertools
x=np.load('x1.npy')[:50000]; h=np.load('h1.npy')[:50000]
M=np.maximum.accumulate(x,1); m=M[:,9]; am=np.argmax(x,1); T=m[:,None]-x
I=np.eye(10)[am]
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
y=h[:,14]
for extra in [[],[T[:,4]],[T[:,4],x[:,9]],[T[:,4],T[:,6]]]:
    B=np.c_[m[:,None]*I,np.array(extra).T] if extra else m[:,None]*I
    r,c=fit(B,y); print(len(extra),'%.3f'%r,c.round(1))
# T4 interactions with am
B=np.c_[m[:,None]*I,T[:,4:5]*I]; r,c=fit(B,y); print('T4*am %.3f'%r,c.round(1))
B=np.c_[m[:,None]*I,T[:,4:5]*I,T]; r,c=fit(B,y); print('+T %.3f'%r,c.round(1))
