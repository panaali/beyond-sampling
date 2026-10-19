import numpy as np, itertools
x=np.load('x1.npy')[:30000]; h=np.load('h1.npy')[:30000]
M=np.maximum.accumulate(x,1); m=M[:,9]; s=np.argsort(-x,1); am=s[:,0]; age=9-am
T=m[:,None]-x; odd=(age%2==1)*1.; G9=np.maximum(x[:,9]-M[:,8],0)
F={'M8':M[:,8],'M9':m,'G9':G9,'M9odd':m*odd,'M9even':m*(1-odd),'negsum':np.maximum(-x,0).sum(1),'sumx':x.sum(1),
   'xsE':x[:,0::2].sum(1),'xsO':x[:,1::2].sum(1),'TsE':T[:,0::2].sum(1),'TsO':T[:,1::2].sum(1),'D9':M[:,8]-x[:,9],'P9':np.maximum(M[:,8]-x[:,9],0)}
for j in range(10):
    F[f'T{j}']=T[:,j]; F[f'x{j}']=x[:,j]; F[f'M{j}']=M[:,j]; F[f'Mam{j}']=m*(am==j)
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
for i in range(16):
    y=h[:,i]; FF=dict(F)
    r,c=fit(np.c_[m,G9,T],y); w=c[2:]; o=np.argsort(w)
    for cuts in [(3,7),(2,8),(1,9),(5,),(8,),(9,),(1,),(2,),(4,)]:
        for g in np.split(o,cuts): FF['T{'+','.join(map(str,sorted(g)))+'}']=T[:,g].sum(1)
    names=list(FF); A=np.stack([FF[k] for k in names],1)
    sc=lambda sel:fit(A[:,sel],y)[0]
    sel=[]
    for step in range(5): sel.append(max((k for k in range(len(names)) if k not in sel),key=lambda k:sc(sel+[k])))
    imp=True
    while imp:
        imp=False; cur=sc(sel)
        for p in range(5):
            for k in range(len(names)):
                if k in sel: continue
                t=sel.copy(); t[p]=k; rr=sc(t)
                if rr>cur+1e-4: sel=t; cur=rr; imp=True
    rr,cc=fit(A[:,sel],y)
    print(i,'%.3f'%rr,' '.join(f'{names[k]}:{v:.2f}' for k,v in zip(sel,cc)),flush=True)
