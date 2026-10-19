import numpy as np
from featlib import *
from feats import fit_relu, fit_gated
np.set_printoptions(precision=2,suppress=True)
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
o=np.argsort(-x,1); a=o[:,0]; b=o[:,1]; s=-np.sort(-x,1)
def show(i,nm,X,g=None):
    X=np.column_stack(X)
    r,c=fit_relu(X,H[:,i],20) if g is None else fit_gated(X,H[:,i],g>0.5,20)
    print(i,nm,'%.3f'%r,np.round(c,2))
P=[3,5,6,7]
g30=((a==8)&np.isin(b,P))|((b==8)&np.isin(a,P))
show(30,'g:[J1_8,s2-s3]',[J1(x,8),s[:,1]-s[:,2]],g30*1.)
show(30,'g:[J1_8,s1-s2,s2-s3]',[J1(x,8),s[:,0]-s[:,1],s[:,1]-s[:,2]],g30*1.)
show(30,'g:[J1_8,x9,s2-s3]',[J1(x,8),x[:,9],s[:,1]-s[:,2]],g30*1.)
g17=(~np.isin(a,[7,8]))&(b!=8)
show(17,'g:[a in 1246, b==7]',[np.isin(a,[1,2,4,6])*1.,(b==7)*1.],g17*1.)
show(17,'g:[a in 1246, b==7, s1-s2]',[np.isin(a,[1,2,4,6])*1.,(b==7)*1.,s[:,0]-s[:,1]],g17*1.)
show(17,'g:[a in 246, b==7, a==9]',[np.isin(a,[2,4,6])*1.,(b==7)*1.,(a==9)*1.],g17*1.)
g11=~np.isin(a,[1,3,6,7])
show(11,'g:[a in 2 8, b==8]',[np.isin(a,[2,8])*1.,(b==8)*1.],g11*1.)
show(11,'g:[a in 2 8, b==8, s1-s2]',[np.isin(a,[2,8])*1.,(b==8)*1.,s[:,0]-s[:,1]],g11*1.)
show(11,'g:[a==2,a==8,b==8,b in 367]',[(a==2)*1.,(a==8)*1.,(b==8)*1.,np.isin(b,[3,6])*1.],g11*1.)
g10=(a!=8)&(b!=8)
show(10,'g:[a,b]',[a*1.,b*1.],g10*1.)
show(10,'g:[a,b,x8,So]',[a*1.,b*1.,x[:,8],So9(x,8)],g10*1.)
show(10,'g:[a,x8,S9]',[a*1.,x[:,8],topk(x,9,1)],g10*1.)
