from lib import *
import sys
for i in map(int,sys.argv[1:]):
    res=[]
    for k in [1,2,3,4,5]:
        pass
    g,r=greedy(h[:,i],B2,names2,k=5)
    print(i,'%.3f'%r,g,flush=True)
