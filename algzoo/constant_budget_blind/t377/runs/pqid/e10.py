import numpy as np, itertools
x=np.load('x2.npy')[:60000];h=np.load('h2.npy')[:60000]
xt=np.load('x2.npy')[100000:160000];ht=np.load('h2.npy')[100000:160000]
def fe(x):
    s7=-np.sort(-x[:,:8],1); s=-np.sort(-x,1); s9=-np.sort(-x[:,:9],1)
    return {'x8':x[:,8],'x9':x[:,9],'M7':s7[:,0],'S7':s7[:,1],'T7':s7[:,2],'mean7':x[:,:8].mean(1),
            'M':s[:,0],'S':s[:,1],'T':s[:,2],'M8':s9[:,0],'S8':s9[:,1],'T8':s9[:,2],'max9_7':np.maximum(x[:,9],s7[:,0])}
F=fe(x);Ft=fe(xt)
def fit(X,y):
    g=y>0.3
    for it in range(6):
        c,*_=np.linalg.lstsq(X[g],y[g],rcond=None); g=(y>0.3)|(X@c>0)
    return c
keys=list(F)
for i in [5,9]:
    res=[]
    for k in itertools.combinations(keys,4):
        X=np.c_[[F[a] for a in k]].T; X=np.c_[X,np.ones(len(x))]
        c=fit(X,h[:,i]); Xt=np.c_[np.c_[[Ft[a] for a in k]].T,np.ones(len(xt))]
        r=1-((ht[:,i]-np.maximum(Xt@c,0))**2).mean()/ht[:,i].var(); res.append((r,k,c))
    res.sort(key=lambda t:-t[0])
    for r,k,c in res[:4]: print(i,'%.4f'%r,k,np.round(c,3))
