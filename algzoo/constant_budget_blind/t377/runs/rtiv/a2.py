import numpy as np
d=np.load('d1.npz');x=d['x'];H=d['H']
s=-np.sort(-x,1)
pm=np.maximum.accumulate(x,1)
def r2(F,y):
    F=np.c_[F,np.ones(len(y))];c,*_=np.linalg.lstsq(F,y,rcond=None);e=y-F@c;return 1-e.var()/y.var(),c
np.set_printoptions(precision=2,suppress=True,linewidth=200)
for i in range(16):
    y=H[:,i]
    out=[]
    for nm,F in [("x",x),("sorted",s),("pm",pm),("x+pm",np.c_[x,pm]),("x+s",np.c_[x,s]),("x+s+pm",np.c_[x,s,pm])]:
        out.append(f"{nm}:{r2(F,y)[0]:.3f}")
    print(i," ".join(out))
print("coef x(0..9) | pm(0..9) | const")
for i in range(16):
    _,c=r2(np.c_[x,pm],H[:,i]); print(i,c[:10],"|",c[10:20],"|",round(c[20],1))
