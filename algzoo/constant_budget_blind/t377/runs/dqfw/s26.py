from lib import *
import sys
for i,p,k in [(24,5,2),(30,4,2),(25,8,2),(28,8,1),(20,0,2)]:
    G=(rank[:,p]>=k)*1.; xp=x[:,p]
    cand={'m1':G*m1,'m2':G*m2,'m3':G*m3,'xp':G*xp,'m1*am':G*m1*am/9,'m2*am':G*m2*am/9,'m1*a2':G*m1*a2/9,'m2*a2':G*m2*a2/9,
          'm1*evam':G*m1*(am%2==0),'m2*evam':G*m2*(am%2==0),'m1*eva2':G*m1*(a2%2==0),'m2*eva2':G*m2*(a2%2==0),'sum':G*x.sum(1),'r3':G*(rank[:,p]==2)*xp}
    nm=list(cand); A=np.stack(list(cand.values()),1)
    g,r=greedy(h[:,i],A,nm,k=5,sub=20000); print(i,p,k,'%.3f'%r,g,flush=True)
