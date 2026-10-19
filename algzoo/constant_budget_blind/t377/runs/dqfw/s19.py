from lib import *
import sys
x9=x[:,9]; M8=rm[:,8]; S8=S[:,8]
which=sys.argv[1]
if which=='a':
  for i in [5,31,17,8]:
    for key,kn in [(am,'am'),(a2,'a2')]:
        bf=lambda sl:[M8[sl],S8[sl],relu(x9-S8)[sl]]
        mf=lambda ip,sl:[ip*M8[sl],ip*S8[sl]]
        P,c,r=psearch(h[:,i],bf,mf,key=key); print(i,kn,P,np.round(c,3),'%.4f'%r,flush=True)
else:
  for i,p,k in [(24,5,2),(30,4,2),(25,8,2),(28,8,1),(20,0,2)]:
    xp=x[:,p]; G=(rank[:,p]>=k)*1.
    bf=lambda sl:[G[sl]*m1[sl],G[sl]*m2[sl],G[sl]*xp[sl]]
    mf=lambda ip,sl:[G[sl]*ip*m1[sl],G[sl]*ip*m2[sl]]
    for key,kn in [(am,'am'),(a2,'a2')]:
        P,c,r=psearch(h[:,i],bf,mf,key=key); print(i,kn,P,np.round(c,3),'%.4f'%r,flush=True)
