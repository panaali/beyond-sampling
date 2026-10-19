from lib import *
def oth(p):  # position of the other top-2 element (when p is in top2)
    return np.where(am==p,a2,am)
x9=x[:,9]
# 21
y=h[:,21]; t=(rank[:,9]<=1)*1.; ot=oth(9); ev=(ot%2==0)*1.
other=np.where(am==9,m2,m1)
for nm,cols in [('base',[relu(x9-m3),relu(x9-m3)*ev]),
                ('x9,oth,m3 ev',[t*x9,t*m3,t*ev*x9,t*ev*m3]),
                ('x9,oth,m3,ev',[t*x9,t*other,t*m3,t*ev*x9,t*ev*m3]),
                ('r+evx9',[relu(x9-m3),relu(x9-m3)*ev,t*ev*m3,t*m3])]:
    c,r=relufit(y,np.stack(cols,1)); print(21,nm,'%.3f'%r,np.round(c,2))
# 14
y=h[:,14]; t=(rank[:,8]<=1)*1.; ot=oth(8); g=(am==8)*1.
for nm,cols in [('rank',[g*x[:,8],(rank[:,8]==1)*x[:,8],(rank[:,8]==2)*x[:,8],(rank[:,8]<=2)*m4]),
                ('t*x8,t*x8*ot',[t*x[:,8],t*x[:,8]*ot]),
                ('t*x8,t*x8*ot,g',[t*x[:,8],t*x[:,8]*ot,g*x[:,8],g*x[:,8]*ot]),
                ('t*x8,t*m3,ot',[t*x[:,8],t*m3,t*x[:,8]*ot,t*m3*ot])]:
    c,r=relufit(y,np.stack(cols,1)); print(14,nm,'%.3f'%r,np.round(c,2))
# 4
y=h[:,4]; e2=(a2%2==0)*1.; e1=(am%2==0)*1.
for nm,cols in [('e2*(m2-m3)',[e2*(m2-m3)]),('e2*m2,e2*m3',[e2*m2,e2*m3]),('e2 m2 m3, e1 m1',[e2*m2,e2*m3,e1*m1, e1*m2]),('e2 not9',[e2*m2*(am!=9),e2*m3*(am!=9),(am!=9)*m2])]:
    c,r=relufit(y,np.stack(cols,1)); print(4,nm,'%.3f'%r,np.round(c,2))
