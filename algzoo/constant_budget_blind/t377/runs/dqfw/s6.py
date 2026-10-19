from lib import *
import sys
def oth(p):
    xo=np.delete(x,p,1); so=-np.sort(-xo,1)
    bef=x[:,:p].max(1) if p>0 else np.full(N,-9.)
    aft=x[:,p+1:].max(1) if p<9 else np.full(N,-9.)
    return so,bef,aft
for i,p in [(2,3),(15,1),(19,6),(22,7),(26,2),(14,8),(10,4),(10,5),(11,4),(11,0),(27,2),(27,6)]:
    y=h[:,i]; so,bef,aft=oth(p); xp=x[:,p]
    out=[]
    for nm,cols in [('x-max',[xp,so[:,0]]),('x-2nd',[xp,so[:,1]]),('x-bef-aft',[xp,bef,aft]),('x-bef-aft-2nd',[xp,bef,aft,so[:,1]])]:
        A=np.stack(cols,1); c,r=relufit(y,A); out.append('%s %.3f %s'%(nm,r,np.round(c,2)))
    print(i,p,' | '.join(out))
