from lib import *
exec(open('s20.py').read().split('for i in map')[0])
rmin=np.minimum.accumulate(x,1)
nm4=names3+['s%d'%j for j in range(10)]+['mn%d'%j for j in range(1,10)]+['absx%d'%j for j in range(10)]+['sum','abssum']
B4=np.concatenate([B3,s,rmin[:,1:],abs(x),x.sum(1,keepdims=True),abs(x).sum(1,keepdims=True)],1)
import sys
for i in map(int,sys.argv[1:]):
    y=h[:,i]
    print(i,'corr min %.2f abs-sum %.2f'%(np.corrcoef(y,s[:,9])[0,1],np.corrcoef(y,abs(x).sum(1))[0,1]))
    greedy_lin(y,B4,nm4,k=5)
