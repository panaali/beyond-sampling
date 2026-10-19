from lib import *
import sys
i=int(sys.argv[1])
y=h[:,i]
for p in range(10):
    for q in [None]:
        g=(am==p)
        A=np.concatenate([x[g],m2[g,None],m3[g,None]],1)
        c,r=relufit(y[g],A)
        print('am=%d n=%d mean %.2f frac>0 %.2f r2 %.3f'%(p,g.sum(),y[g].mean(),(y[g]>0).mean(),r),' '.join('%+.2f'%v for v in c))
