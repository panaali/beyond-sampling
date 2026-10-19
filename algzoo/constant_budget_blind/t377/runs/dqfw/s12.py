from lib import *
import sys
i=int(sys.argv[1]); p=int(sys.argv[2])
nm,A=pbank2(p)
for k in [3,5]:
    g,r=greedy(h[:,i],A,nm,k=k,sub=15000); print(i,p,k,'%.3f'%r,g,flush=True)
