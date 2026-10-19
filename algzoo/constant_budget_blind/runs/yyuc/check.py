import numpy as np, labels
x=np.load('x7.npy'); H=np.load('H7.npy')
r=[1-((labels.f(i,x)-H[:,i])**2).mean()/H[:,i].var() for i in range(16)]
print(np.round(r,4), 'mean', np.mean(r)); print(labels.CONSTANTS)
x2=np.random.default_rng(1).standard_normal((1000,10)); print([labels.f(i,x2).shape for i in range(16)])
s=open('labels.py').read(); print([b for b in ['torch','alg_zoo','weight_hh','weight_ih','state_dict','.pth','probe'] if b in s])
