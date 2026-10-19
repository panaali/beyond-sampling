import numpy as np, feats, probe, json
np.set_printoptions(precision=3,suppress=True,linewidth=230)
rng=np.random.default_rng(16)
x=rng.standard_normal((40000,10)); H=probe.hidden(x)
xt=rng.standard_normal((40000,10)); Ht=probe.hidden(xt)
R=lambda z:np.maximum(z,0)
def featT(x,t,T):
    pre=x[:,:t]; q=-np.sort(-pre,1); am=pre.argmax(1)
    D=np.isin(am,T)*(q[:,0]-q[:,1])
    return np.c_[x[:,t],q[:,0],q[:,1],D,np.ones(len(x))]
out={}
for i,t,T in [(14,8,[1,2,3]),(27,8,[0]),(24,7,[0,1]),(26,7,[0]),(23,6,[0])]:
    c=feats.relufit(H[:,i],featT(x,t,T)); p=R(featT(xt,t,T)@c)
    r=1-((Ht[:,i]-p)**2).mean()/Ht[:,i].var()
    print(i,t,T,c,'test R2 %.3f'%r); out[i]=dict(t=t,T=T,c=[float(v) for v in c],r2=float(r))
json.dump(out,open('fit_tfam.json','w'))
