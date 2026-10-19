from lib import *
sx=x.sum(1)
for i,p in [(2,3),(15,1),(19,6),(22,7),(26,2),(14,8)]:
    y=h[:,i]; xo=np.delete(x,p,1); so=-np.sort(-xo,1); xp=x[:,p]
    for nm,cols in [('xp,sum',[xp,sx]),('xp,sum,maxo',[xp,sx,so[:,0]]),('xp,sum,m1,m2',[xp,sx,m1,m2]),('xp,sum,maxo,2ndo',[xp,sx,so[:,0],so[:,1]])]:
        A=np.stack(cols,1); c,r=relufit(y,A); print(i,p,nm,'%.3f'%r,np.round(c,2))
