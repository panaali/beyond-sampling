import numpy as np, json; from lib2 import fitrelu,R
d=np.load('d1.npz'); x=d['x'][:30000].astype(float); h=d['h'][:30000]
y=x[:,:9]; s=np.sort(y,1)[:,::-1]; a=np.argsort(-y,1); A,B,C=s[:,0],s[:,1],s[:,2]; pA,pB=a[:,0],a[:,1]; x9=x[:,9]
X=np.c_[[A*(pA==j) for j in range(9)]+[B*(pB==j) for j in range(9)]+[x9,C]].T
def split(v):
    o=np.argsort(v); best=None
    for k in range(1,9):
        lo,hi=v[o[:k]],v[o[k:]]; e=((lo-lo.mean())**2).sum()+((hi-hi.mean())**2).sum()
        if best is None or e<best[0]: best=(e,k)
    k=best[1]; lo,hi=o[:k],o[k:]
    return sorted((lo if len(lo)<=len(hi) else hi).tolist())
out={}
for i in range(32):
    r2f,c=fitrelu(X,h[:,i],8)
    SA=split(c[:9]); SB=split(c[9:18])
    inA=np.isin(pA,SA); inB=np.isin(pB,SB)
    cands={}
    base=[A,B,A*inA,B*inB]
    for nm,last in [('x9',x9),('C',C)]:
        Z=np.c_[base+[last]].T; r2,cc=fitrelu(Z,h[:,i],8); cands[nm]=(r2,cc)
    nm=max(cands,key=lambda k:cands[k][0]); r2,cc=cands[nm]
    print(i,'full %.3f'%r2f,'fam %.3f'%r2,'SA',SA,'SB',SB,'last',nm,'coef[A,B,A@SA,B@SB,last]',np.round(cc,2))
    out[i]=dict(SA=SA,SB=SB,last=nm,c=cc.tolist(),r2=r2)
json.dump(out,open('fam.json','w'))
