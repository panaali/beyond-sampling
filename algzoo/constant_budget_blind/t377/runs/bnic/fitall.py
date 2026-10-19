import numpy as np, importlib, sys, probe
src=open('labels_template.py').read()
ns={}; exec(src.replace('__COEFS__','None'),ns)
x=np.load('x1.npy'); h=np.load('h1.npy')
rng=np.random.default_rng(7); xt=rng.standard_normal((100000,10)); ht=probe.hidden(xt); np.save('xt.npy',xt); np.save('ht.npy',ht)
coefs={}
for i in range(16):
    B=np.stack(ns['_terms'](i,x),1); c,*_=np.linalg.lstsq(B,h[:,i],rcond=None); coefs[i]=[float('%.4g'%v) for v in c]
ns['COEFS']=coefs
for i in range(16):
    p=ns['f'](i,xt); r=1-((ht[:,i]-p)**2).mean()/ht[:,i].var(); print(i,'test R2 %.3f'%r, coefs[i])
open('labels_fit.py','w').write(src.replace('__COEFS__',repr(coefs)))
