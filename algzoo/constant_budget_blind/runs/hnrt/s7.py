import numpy as np, probe
from util import *
rng=np.random.default_rng(6)
x=rng.standard_normal((50000,10)); H=probe.hidden(x)
x9=x[:,9]; o=np.argsort(-x[:,:9],1); s=np.take_along_axis(x[:,:9],o,1); p=o[:,0]; m=s[:,0]
for i,S in [(0,[0,2,3,5,6]),(3,[1,2,4,5]),(6,[0,1,4,7])]:
    h=H[:,i]; ok=(p!=8)
    inS=np.isin(p,S)
    F=np.stack([m*inS*ok, m*(~inS)*ok, s[:,1]*ok, x9*ok],1)
    w=relufit(F,h); print(i,np.round(w,2),round(r2(np.maximum(F@w,0),h),4))
    F=np.stack([m*inS*ok, m*(~inS)*ok, x9*ok],1)
    w=relufit(F,h); print(i,np.round(w,2),round(r2(np.maximum(F@w,0),h),4))
h=H[:,10]
F=np.stack([x9,s[:,1],s[:,2]],1); w=relufit(F,h); print(10,np.round(w,2),round(r2(np.maximum(F@w,0),h),4))
F=np.stack([x9,s[:,1]+s[:,2]],1); w=relufit(F,h); print(10,np.round(w,2),round(r2(np.maximum(F@w,0),h),4))
