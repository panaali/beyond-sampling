import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy');H=np.load('H1.npy')
rk=np.argsort(np.argsort(-x,1),1) # rank of each position, 0=largest
rk9=np.argsort(np.argsort(-x[:,:9],1),1)
for i,j in [(27,0),(25,1),(29,1),(0,2),(28,3),(14,3),(4,4),(13,5),(7,6),(24,7),(3,8),(10,8),(11,8),(17,6),(21,4),(30,8)]:
    m=[H[rk[:,j]==r,i].mean() for r in range(10)]
    s=[H[rk[:,j]==r,i].std() for r in range(10)]
    m9=[H[rk9[:,j]==r,i].mean() for r in range(9)]
    print(i,j,'mean by rank',np.array(m),' sd',np.array(s)); print('      by rank in 0..8',np.array(m9))
