from lib import *
import sys
for i in range(32):
    y=h[:,i]
    cl,rl=fit(y,list(Abank.T))
    c,r=relufit(y,Abank)
    top=np.argsort(-abs(c))[:7]
    print(i,'lin %.3f relu %.3f'%(rl,r),' '.join('%s:%+.2f'%(names[k],c[k]) for k in top))
