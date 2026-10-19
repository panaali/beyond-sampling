import numpy as np
d=np.load('d1.npz');x=d['x'];H=d['H']
o=np.argsort(-x,1);am=o[:,0];a2=o[:,1]
T=np.arange(10)[None]
pm=np.maximum.accumulate(x,1)
pre=x*(T<am[:,None]);post=x*(T>am[:,None]);at=x*(T==am[:,None])
cap=np.c_[x[:,:1],np.minimum(x[:,1:],pm[:,:-1])]
up=np.c_[np.zeros(len(x)),np.maximum(x[:,1:]-pm[:,:-1],0)]
def r2(F,y):
    c,*_=np.linalg.lstsq(F,y,rcond=None);return 1-((y-F@c)**2).mean()/y.var(),c
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in range(16):
    y=H[:,i]
    print(i,"prepostat %.3f"%r2(np.c_[pre,post,at],y)[0],"cap+up %.3f"%r2(np.c_[cap,up],y)[0],"cap+pm %.3f"%r2(np.c_[cap,pm],y)[0],"x+pm+pre %.3f"%r2(np.c_[x,pm,pre],y)[0])
print("cap(0..9) | up(1..9)")
for i in range(16):
    _,c=r2(np.c_[cap,up[:,1:]],H[:,i]); print("%2d"%i,c[:10],"|",c[10:])
