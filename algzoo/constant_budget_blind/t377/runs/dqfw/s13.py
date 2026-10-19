from lib import *
m4=s[:,3]
rank=np.argsort(o,1)  # rank[n,j] = rank of position j (0=max)
for i,p in [(22,7),(19,6),(2,3),(15,1),(26,2),(14,8),(8,5),(10,4),(10,5),(11,4),(11,0),(11,6),(27,2),(27,6),(3,6),(3,7)]:
    y=h[:,i]; xp=x[:,p]; rk=rank[:,p]
    r1=(rk==0)*1.;r2_=(rk==1)*1.;r3=(rk==2)*1.;t3=(rk<=2)*1.
    c1,q1=fit(y,[r1*xp,r2_*xp,r3*xp,t3*m4])
    c2,q2=fit(y,[relu(xp-m4)*t3,r1*m4,r2_*m4])
    c3,q3=fit(y,[r1*xp,r2_*xp,r3*xp,t3*m4,(rk>2)*relu(xp)])
    print(i,p,'rankxp+m4 %.3f'%q1,np.round(c1,2),' greedyform %.3f'%q2,np.round(c2,2),' +rest %.3f'%q3)
