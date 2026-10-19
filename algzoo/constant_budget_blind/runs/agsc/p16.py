import numpy as np, probe
from feat import r2
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(15); N=100000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
def lin(A,y):
    A=np.c_[A,np.ones(len(y))]; c=np.linalg.lstsq(A,y,rcond=None)[0]; return r2(y,A@c),c
for i,p in [(1,6),(6,4),(0,5),(7,5),(2,5),(3,5)]:
    for lo,hi in [(-9,-1),(-0.5,0.5),(1,9)]:
        m=(x[:,p]>lo)&(x[:,p]<hi); y=h[m,i]
        pre=np.sort(x[m,:p],1)[:,::-1][:,:3]
        post=x[m,p:]
        r_pre,_=lin(pre,y); r_post,_=lin(post,y); r_all,c=lin(np.c_[pre,post],y)
        print(i,'p',p,'x_p in',(lo,hi),'mean',round(y.mean(),2),'R2 pre-top3',round(r_pre,2),'post',round(r_post,2),'both',round(r_all,2),c[:-1])
