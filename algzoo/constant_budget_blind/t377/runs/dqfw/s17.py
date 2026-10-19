from lib import *
for i,p,k in [(24,5,2),(30,4,1),(30,4,2),(25,8,2),(28,8,1),(20,0,1),(20,0,2)]:
    y=h[:,i]; xp=x[:,p]; rk=rank[:,p]; mk=s[:,k-1]; G=(rk>=k)*1.
    c1,q1=fit(y,[relu(mk-xp)])
    c2,q2=fit(y,[G*m1,G*m2,G*xp])
    c3,q3=fit(y,[G*m1,G*m2,G*m3,G*xp,G*m4])
    c4,q4=fit(y,[relu(mk-xp),G*m1,G*m2])
    print(i,p,k,'U1 %.3f'%q1,np.round(c1,2),'U2 %.3f'%q2,np.round(c2,2),'U3 %.3f'%q3,np.round(c3,2),'U4 %.3f'%q4,np.round(c4,2))
