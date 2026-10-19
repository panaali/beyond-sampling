import numpy as np, probe
from lib import fit_relu
from lib2 import library2
rng=np.random.default_rng(3)
x=rng.standard_normal((60000,10)); h=probe.hidden(x)
np.save('x3.npy',x); np.save('h3.npy',h)
F=library2(x)
F['amHi']=((F['am7']+F['am8'])>0)*1.
def t(i,fs):
    X=np.stack([F[f] for f in fs],1); b,r=fit_relu(X,h[:,i],it=15)
    print(i,fs,np.round(b,3),'R2 %.4f'%r)
t(22,['x9','P8']); t(31,['x9','Q8']); t(6,['x9','Q8']); t(6,['x9','Q8','P8'])
t(9,['x8','P7']); t(26,['P8','Q8','am8']); t(26,['P8','P7','Q8'])
t(12,['P8','Q8']); t(12,['P8','Q8','am0']); t(5,['P8','Q8']); t(5,['P8','Q8','am0'])
t(19,['x9','P8','Q8']); t(18,['x9','P8','Q8']); t(18,['x9','P8','Q8','amHi'])
for i in [1,15,8,2,20]: t(i,['x9','P8','Q8'])
