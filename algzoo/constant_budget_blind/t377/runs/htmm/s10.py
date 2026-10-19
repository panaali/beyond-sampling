import numpy as np,sys
x=np.load('x2.npy'); h=np.load('h2.npy')
p=x[:,:9]; o=np.argsort(-p,1); rk=np.argsort(o,1)
for i,k in [(0,2),(7,6),(27,0),(4,4),(13,5),(25,1)]:
    y=h[:,i]
    for g in [0,1]:
        m=rk[:,k]==g
        other=o[:,1] if g==0 else o[:,0]
        print(i,k,'rank',g,'by pos of other top2:',' '.join('%d:%.2f'%(j,y[m&(other==j)].mean()) for j in range(9) if j!=k))
    m=rk[:,k]==2
    print(i,k,'rank2 by (am,a2) mean - tops:', ' '.join('%d%d:%.1f'%(a,b,y[m&(o[:,0]==a)&(o[:,1]==b)].mean()) for a in range(9) for b in range(9) if a!=b and a!=k and b!=k and (m&(o[:,0]==a)&(o[:,1]==b)).sum()>30)[:400])
