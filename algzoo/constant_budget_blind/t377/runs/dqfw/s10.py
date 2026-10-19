from lib import *
import sys
i=int(sys.argv[1]); p=int(sys.argv[2])
nm,A=pbank(p)
for k in [2,3,4]:
    g,r=greedy(h[:,i],A,nm,k=k,sub=20000); print(i,p,k,'%.3f'%r,g,flush=True)
