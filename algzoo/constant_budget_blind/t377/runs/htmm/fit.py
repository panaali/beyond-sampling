import numpy as np,sys,json
from lib import fit_relu
from fe import base,feat
xa=np.load('x2.npy'); ha=np.load('h2.npy'); xb=np.load('x3.npy'); hb=np.load('h3.npy')
Da=base(xa); Db=base(xb)
specs=[a.split('=') for a in sys.argv[1:]]
for i,fs in specs:
    i=int(i); fs=fs.split(',')
    Xa=np.stack([feat(Da,f) for f in fs],1); b,r=fit_relu(Xa,ha[:,i],it=20)
    Xb=np.stack([feat(Db,f) for f in fs],1); pr=np.maximum(np.c_[Xb,np.ones(len(xb))]@b,0)
    r2=1-((pr-hb[:,i])**2).mean()/hb[:,i].var()
    print(i,fs,'b=',np.round(b,3).tolist(),'testR2 %.4f'%r2,flush=True)
