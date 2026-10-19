from lib import *
sx=x.sum(1)
for i,p in [(2,3),(15,1),(19,6),(22,7)]:
    y=h[:,i]; xp=x[:,p]; rk=rank[:,p]
    r1=(rk==0)*1.;r2_=(rk==1)*1.;r3=(rk==2)*1.;t3=(rk<=2)*1.
    so=sx-xp
    sa=x[:,p+1:].sum(1)
    for nm,cols in [('base',[r1*xp,r2_*xp,r3*xp,t3*m4]),('+so',[r1*xp,r2_*xp,r3*xp,t3*m4,t3*so]),('so no m4',[r1*xp,r2_*xp,r3*xp,t3*so]),('+sa',[r1*xp,r2_*xp,r3*xp,t3*m4,t3*sa]),('xp,so,relu',[r1*xp,r2_*xp,r3*xp,so,m4])]:
        c,r=relufit(y,np.stack(cols,1)); print(i,p,nm,'%.3f'%r,np.round(c,2))
