import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
G=np.load('G2.npy');x=np.load('x2.npy');H=np.load('H2.npy')
P=np.maximum.accumulate(x,1)
for i,j in [(27,0),(25,1),(29,1),(0,2),(28,3),(4,4),(13,5),(7,6),(24,7),(3,8),(16,4),(23,7)]:
    a=H[:,i]>1e-4
    rec=np.ones(len(x),bool) if j==0 else x[:,j]>P[:,j-1]
    ismax=x[:,j]>=P[:,9]
    for nm,m in [('rec',rec&a),('norec',(~rec)&a),('max',ismax&a)]:
        if m.sum()<20: continue
        g=G[m,i,:]
        print(i,j,nm,'n=%d act=%.2f'%(m.sum(), (a&(m|True))[rec if nm=='rec' else (~rec if nm=='norec' else ismax)].mean()),np.median(g,0),'sd',g.std(0).round(1))
