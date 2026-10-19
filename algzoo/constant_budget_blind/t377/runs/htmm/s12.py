import numpy as np
x=np.load('x2.npy'); h=np.load('h2.npy')
p=x[:,:9]; o=np.argsort(-p,1)
np.set_printoptions(precision=1,suppress=True,linewidth=200)
for i in [11,17,30,21]:
    T=np.full((9,9),np.nan)
    for a in range(9):
        for b in range(9):
            m=(o[:,0]==a)&(o[:,1]==b)
            if m.sum()>20: T[a,b]=h[m,i].mean()
    print('neuron',i,'rows=am cols=a2 (prefix x0..x8)'); print(T)
    r3=o[:,2]; print(' by 3rd pos:',' '.join('%.2f'%h[r3==k,i].mean() for k in range(9)))
