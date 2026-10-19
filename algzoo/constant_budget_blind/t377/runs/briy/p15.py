import numpy as np
from dic3 import relufit, pref
from dic4 import J
x=np.load('x1.npy')[:30000]; h=np.load('h1.npy')[:30000]
s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
M=np.maximum.accumulate(x,1); M8,S8,A8=pref(x,8); M7,S7,A7=pref(x,7)
od8=(A8%2==1); od7=(A7%2==1)
X=lambda k:x[:,k]
forms={
 '12a':(12,[X(3),X(1),M8,S8]),
 '12b':(12,[X(3),X(1),M[:,2],s[:,1]]),
 '12c':(12,[X(3),X(1),np.minimum(X(3),M[:,2]),s[:,1],X(0)]),
 '12d':(12,[X(3),X(1),s[:,0],s[:,1],s[:,2]]),
 '15a':(15,[X(2),X(5),M8,M8*od8,X(0)]),
 '15b':(15,[X(2),X(5),s[:,0],s[:,1],s[:,0]*(am%2==0)]),
 '15c':(15,[X(2),X(5),M8,S8,X(1)]),
 '3a':(3,[X(2),X(1),X(4),M7,M7*od7]),
 '3b':(3,[X(2),X(1),X(4),s[:,0],s[:,1]]),
 '14a':(14,[X(6)*(am==6),s[:,1]*(am==6),X(9)*(am==6)]),
 '14b':(14,[X(6),s[:,1],X(9),X(7),X(8)]),
 '9a':(9,[M7,X(5),X(4),X(2)+X(6)+X(8),X(0)]),
 '9b':(9,[M8,X(5),X(4),X(2)+X(6)+X(8),M8*od8]),
}
for k,(i,F) in forms.items():
    c,r=relufit(np.stack(F,1),h[:,i]); print(k,'R2 %.3f'%r,np.round(c,2))
