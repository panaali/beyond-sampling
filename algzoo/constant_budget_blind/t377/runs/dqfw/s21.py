from lib import *
y=h[:,4]
ev=np.arange(10)%2==0
e2=ev[a2]*1.; e1=ev[am]*1.
se=x[:,ev].sum(1); so_=x[:,~ev].sum(1)
for cols,nm in [([e2*m2],'e2*m2'),([e2*m2,e1*m1],'e2m2,e1m1'),([se,so_],'se,so'),([e2*m2,(1-e2)*m2,e1*m1,(1-e1)*m1],'4'),([e2*m2,(1-e2)*m2,e1*m1,(1-e1)*m1,np.ones(N)],'4+1')]:
    c,r=fit(y,cols); print(nm,'%.3f'%r,np.round(c,2))
    c,r=relufit(y,np.stack(cols,1)); print('  relu %.3f'%r,np.round(c,2))
