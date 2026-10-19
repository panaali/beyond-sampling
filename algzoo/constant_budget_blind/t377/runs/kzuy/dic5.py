import numpy as np
from dic4 import dic4
r=lambda z:np.maximum(z,0)
def dic5(x):
    D=dic4(x)
    s=np.sort(x,1)[:,::-1]; am=np.argmax(x,1); g=s[:,0]-s[:,1]
    E=(am%2==0)&(am<8); O=(am%2==1)&(am<8)
    D['PE*g']=E*g; D['PO*g']=O*g; D['A8*g']=(am==8)*g; D['A9*g']=(am==9)*g; D['g']=g
    D['A8*s0']=(am==8)*s[:,0]; D['A9*s0']=(am==9)*s[:,0]
    M8=x[:,:9].max(1); a8=np.argmax(x[:,:9],1)
    D['E8M8']=((a8%2==0)&(a8<8))*M8; D['O8M8']=(a8%2==1)*M8; D['A8M8']=(a8==8)*M8
    D['min(x9,M8)']=np.minimum(x[:,9],M8)
    return D
