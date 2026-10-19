import numpy as np, probe, labels
x=np.random.default_rng(7).standard_normal((50000,10)); h=probe.hidden(x)
r=[]
for i in range(32):
    p=labels.f(i,x); r.append(1-((p-h[:,i])**2).mean()/h[:,i].var())
print(' '.join('%d:%.3f'%(i,v) for i,v in enumerate(r))); print('mean %.3f median %.3f'%(np.mean(r),np.median(r)))
H=np.stack([labels.f(i,x) for i in range(32)],1); print('total var-weighted R2 %.3f'%(1-((H-h)**2).sum()/((h-h.mean(0))**2).sum()))
print(max(labels.CONSTANTS.values()))
s=open('labels.py').read(); print([b for b in ['torch','alg_zoo','weight_hh','weight_ih','state_dict','.pth','probe'] if b in s])
