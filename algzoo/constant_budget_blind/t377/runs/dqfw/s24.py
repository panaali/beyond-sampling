from lib import *
import sys
i=int(sys.argv[1]); p=int(sys.argv[2])
y=h[:,i]
for nm,g in [('am=p',am==p),('a2=p',a2==p),('rank2',rank[:,p]==2),('rest',rank[:,p]>2)]:
    A=np.concatenate([x[g],m1[g,None],m2[g,None],m3[g,None],m4[g,None]],1)
    c,r=relufit(y[g],A)
    print(nm,'n=%d mean %.2f r2 %.3f'%(g.sum(),y[g].mean(),r),' '.join('%+.2f'%v for v in c), 'var share %.2f'%(((y[g]-y.mean())**2).sum()/((y-y.mean())**2).sum()))
