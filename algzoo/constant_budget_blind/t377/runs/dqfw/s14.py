from lib import *
import sys
for i in map(int,sys.argv[1:]):
    g,r=greedy(h[:,i],B3,names3,k=5,sub=12000); print(i,'%.3f'%r,g,flush=True)
