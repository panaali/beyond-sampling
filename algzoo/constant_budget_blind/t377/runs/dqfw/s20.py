from lib import *
import sys
def greedy_lin(y,A,nm,k=5,sub=30000):
    yy=y[:sub]; AA=A[:sub]; sel=[]
    for step in range(k):
        best=None
        for j in range(A.shape[1]):
            if j in sel: continue
            c,r=fit(yy,list(AA[:,sel+[j]].T))
            if best is None or r>best[1]: best=(j,r)
        sel.append(best[0])
        c,r=fit(y,list(A[:,sel].T))
        print('  ',step+1,'%.3f'%r,[(nm[j],round(float(cc),3)) for j,cc in zip(sel,c)],flush=True)
for i in map(int,sys.argv[1:]):
    print(i); greedy_lin(h[:,i],B3,names3,k=6)
