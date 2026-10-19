from lib import *
y=h[:,3]
for q in range(10):
    g=a2==q
    A=np.concatenate([x[g],m1[g,None],m2[g,None],m3[g,None]],1)
    c,r=relufit(y[g],A)
    print('a2=%d mean %.2f r2 %.3f'%(q,y[g].mean(),r),' '.join('%+.2f'%v for v in c))
