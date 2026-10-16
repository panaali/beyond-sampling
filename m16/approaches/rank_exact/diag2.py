"""Diagnostic: per full ordering, how well do fixed-gate linear surrogates reproduce P(correct|ordering)?"""
import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from common import Wi,Wh,Wo
T=10
def gates(x):
    h=np.zeros(16); G=[]
    for t in range(T):
        u=Wh@h+Wi*x[t]; g=u>0; G.append(g); h=u*g
    return G
def linmap(G):
    J=np.zeros((16,T))
    for t in range(T):
        J=Wh@J; J[:,t]+=Wi; J*=G[t][:,None]
    return Wo@J   # logits = A x
def fwd(X):
    h=np.zeros((len(X),16))
    for t in range(T): h=np.maximum(h@Wh.T+X[:,t:t+1]*Wi,0)
    return h@Wo.T
rng=np.random.default_rng(5)
bank=np.sort(rng.standard_normal((4000,T)),1)[:,::-1]   # model-free sorted Gaussians, z1>=z2>=...
EOS=np.sort(np.random.default_rng(0).standard_normal((400000,T)),1)[:,::-1].mean(0)
res=[]
for it in range(400):
    perm=rng.permutation(T)    # perm[r]=position of rank r
    X=np.empty_like(bank); X[:,perm]=bank
    tgt=perm[1]
    truth=(fwd(X).argmax(1)==tgt).mean()
    xc=np.empty(T); xc[perm]=EOS
    A0=linmap(gates(xc))
    s1=((X@A0.T).argmax(1)==tgt).mean()
    # tie-point gates
    x23=xc.copy(); x23[perm[2]]=x23[perm[1]]=(EOS[1]+EOS[2])/2
    x12=xc.copy(); x12[perm[0]]=x12[perm[1]]=(EOS[0]+EOS[1])/2
    A23=linmap(gates(x23)); A12=linmap(gates(x12))
    L0=X@A0.T; L23=X@A23.T; L12=X@A12.T
    m=L0[:,[tgt]]-L0; m[:,tgt]=1; m[:,perm[2]]=(L23[:,tgt]-L23[:,perm[2]]); m[:,perm[0]]=(L12[:,tgt]-L12[:,perm[0]])
    s2=(m.min(1)>0).mean()
    res.append((truth,s1,s2))
r=np.array(res); print('mean truth, S1 canonical gates, S2 tie gates:',r.mean(0))
print('mean abs err per ordering',np.abs(r[:,1:]-r[:,[0]]).mean(0))
