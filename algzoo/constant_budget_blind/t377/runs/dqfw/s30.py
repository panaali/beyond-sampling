from lib import *
for i,p in [(24,5),(30,4),(20,0),(25,8),(28,8)]:
    y=h[:,i]; xp=x[:,p]; res=[]
    for nm,F in [('m1',m1),('m2',m2),('m3',m3)]+[('S%d'%k,S[:,k]) for k in range(1,10)]+[('M%d'%k,rm[:,k]) for k in range(1,10)]:
        f=relu(F-xp); c,r=fit(y,[f]); res.append((r,nm))
    res.sort(reverse=True); print(i,p,[(n,round(r,3)) for r,n in res[:4]])
    # best pair
    r0,n0=res[0]; F0=dict([('m1',m1),('m2',m2),('m3',m3)]+[('S%d'%k,S[:,k]) for k in range(1,10)]+[('M%d'%k,rm[:,k]) for k in range(1,10)])
    best=[]
    for n1 in F0:
        c,r=fit(y,[relu(F0[n0]-xp),relu(F0[n1]-xp),F0[n1]*(rank[:,p]>=2)]); best.append((r,n1,np.round(c,2)))
    best.sort(key=lambda t:-t[0]); print('   pair',n0,best[:2])
