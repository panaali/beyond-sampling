import numpy as np
from dic import dic, r
def dic2(x):
    D=dic(x)
    s=np.sort(x,1)[:,::-1]; am=np.argmax(x,1); a2=np.argsort(-x,1)[:,1]
    M=np.maximum.accumulate(x,1)
    po=(am%2==1)*1.; pe=1-po; qo=(a2%2==1)*1.; qe=1-qo
    D['po*s0']=po*s[:,0]; D['pe*s0']=pe*s[:,0]; D['qo*s1']=qo*s[:,1]; D['qe*s1']=qe*s[:,1]
    D['po*s1']=po*s[:,1]; D['pe*s1']=pe*s[:,1]
    for t in range(10):
        D[f'am{t}*s0']=(am==t)*s[:,0]; D[f'a2{t}*s1']=(a2==t)*s[:,1]
        D[f'po*x{t}']=po*x[:,t]; D[f'pe*x{t}']=pe*x[:,t]
        D[f'pe*(M9-x{t})']=pe*(s[:,0]-x[:,t]); D[f'po*(M9-x{t})']=po*(s[:,0]-x[:,t])
        D[f'r(x{t})']=r(x[:,t]); D[f'r(-x{t})']=r(-x[:,t])
    # parity of max-so-far position at step t
    for t in range(1,10):
        amt=np.argmax(x[:,:t+1],1); D[f'Mpo{t}']=(amt%2==1)*M[:,t]; D[f'Mpe{t}']=(amt%2==0)*M[:,t]
    return D
