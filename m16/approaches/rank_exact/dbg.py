import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from common import Wi,Wh,Wo
from adf import propagate, p_argmax
from diag4 import fwd, mu, S
rng=np.random.default_rng(3); perm=rng.permutation(10); P=np.zeros((10,10)); P[perm,np.arange(10)]=1
m,C=P@mu,P@S@P.T
X=rng.multivariate_normal(m,C,400000); L=fwd(X)
mL,CL=propagate(m,C,Wi,Wh,Wo)
print(np.round(mL,2)); print(np.round(L.mean(0),2))
print(np.round(np.sqrt(np.diag(CL)),2)); print(np.round(L.std(0),2))
tgt=perm[1]; print('MC gauss-input acc',(L.argmax(1)==tgt).mean(),'adf',p_argmax(mL,CL,tgt))
D=-np.eye(10); D[:,tgt]+=1; D=np.delete(D,tgt,0); Md=L@D.T
print('MC margins mean',np.round(Md.mean(0),2),'adf',np.round(D@mL,2))
print('MC P(all>0)',(Md.min(1)>0).mean())
