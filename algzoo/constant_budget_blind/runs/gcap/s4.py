from lib import *
F=np.c_[x,M[:,1:],S[:,1:]]
names=[f'x{t}' for t in range(10)]+[f'M{t}' for t in range(1,10)]+[f'S{t}' for t in range(1,10)]
for i in range(8):
    R,c,p=relufit(F,H[:,i])
    o=np.argsort(-abs(c))[:10]
    print(i,round(R,3),[(names[k],round(c[k],2)) for k in o])
