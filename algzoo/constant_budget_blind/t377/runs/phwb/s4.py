import numpy as np
x=np.load('x0.npy');h=np.load('h0.npy')
P=x[:,:9]; p=P.argmax(1); M=P.max(1); s=np.sort(P,1); m2=s[:,-2]; mean=(P.sum(1)-M)/8
N=len(x); idx=np.arange(9)
after=np.where(idx[None,:]>p[:,None],P,-9).max(1); before=np.where(idx[None,:]<p[:,None],P,-9).max(1)
after=np.maximum(after,-3); before=np.maximum(before,-3)
cands={'m2':m2,'mean':mean,'after':after,'before':before}
for i in [0,10,11,12,14,15,3,13]:
    out=[]
    for q in range(9):
        g=(p==q)&(h[:,i]>0.5)
        row=[]
        for k,v in cands.items():
            A=np.c_[M[g],v[g],np.ones(g.sum())]
            c,_,_,_=np.linalg.lstsq(A,h[g,i],rcond=None)
            row.append(1-((A@c-h[g,i])**2).mean()/h[g,i].var())
        A=np.c_[M[g],m2[g],mean[g],after[g],before[g],np.ones(g.sum())]
        c,_,_,_=np.linalg.lstsq(A,h[g,i],rcond=None)
        row.append(1-((A@c-h[g,i])**2).mean()/h[g,i].var())
        out.append(f"{q}:"+"/".join(f"{r:.2f}" for r in row)+f" c={np.round(c[:5],1)}")
    print(i); print("\n".join(out))
