from lib import *
import sys
sx=x.sum(1)
extra=[];en=[]
for p in range(10):
    for k in [0,1,2]:
        extra.append((rank[:,p]<=k)*sx); en.append('I%d_%d*sum'%(k+1,p))
extra.append(sx); en.append('sum')
A=np.concatenate([B3,np.stack(extra,1)],1); nm=names3+en
for i in map(int,sys.argv[1:]):
    g,r=greedy(h[:,i],A,nm,k=5,sub=12000); print(i,'%.3f'%r,g,flush=True)
