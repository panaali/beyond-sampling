from lib import *
y=h[:,14]; t=(rank[:,8]<=1)*1.; x8=x[:,8]
other=np.where(am==8,m2,m1); s17=x[:,1:8].sum(1); sx=x.sum(1)
for nm,cols in [('A',[x8,m1,m2,m3,s17]),('B',[t*other,t*m3,t*s17]),('C',[t*other,t*m3,t*s17,relu(x8-m3)]),('D',[t*other,t*m3,s17,relu(x8-m3)]),('E',[t*other,t*m3,s17,relu(x8-m3),x8]),('F',[t*other,t*m3,sx,relu(x8-m4)])]:
    c,r=relufit(y,np.stack(cols,1)); print(14,nm,'%.3f'%r,np.round(c,2))
