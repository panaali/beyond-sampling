import numpy as np, sys
x=np.load('x1.npy')[:20000]; h=np.load('h1.npy')[:20000]
srt=-np.sort(-x,1); s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]; m=srt[:,0]; s1=srt[:,1]
M=np.maximum.accumulate(x,1); T=m[:,None]-x; G9=np.maximum(x[:,9]-M[:,8],0)
p8=-np.sort(-x[:,:9],1); S8=p8[:,1]
F={'G9':G9,'M9':m,'gap2':m-s1,'s1':s1,'s2':srt[:,2],'Q9':np.maximum(S8-x[:,9],0),'R9':np.maximum(x[:,9]-S8,0),'S8':S8,'M8':M[:,8],
   'M9amE':m*(am%2==0),'gap2amE':(m-s1)*(am%2==0),'gap2a2E':(m-s1)*(a2%2==0),'s1a2E':s1*(a2%2==0)}
for j in range(10):
    F[f'T{j}']=T[:,j]; F[f'Mam{j}']=m*(am==j); F[f'g2a2_{j}']=(m-s1)*(a2==j); F[f'g2am{j}']=(m-s1)*(am==j); F[f's{j}']=srt[:,j]
names=list(F); A=np.stack([F[k] for k in names],1)
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
for i in [int(a) for a in sys.argv[1:]]:
    y=h[:,i]; sc=lambda sel:fit(A[:,sel],y)[0]; sel=[]
    for st in range(5): sel.append(max((k for k in range(len(names)) if k not in sel),key=lambda k:sc(sel+[k])))
    imp=True
    while imp:
        imp=False; cur=sc(sel)
        for p in range(5):
            for k in range(len(names)):
                if k in sel: continue
                t=sel.copy(); t[p]=k; rr=sc(t)
                if rr>cur+1e-4: sel=t; cur=rr; imp=True
    r,c=fit(A[:,sel],y); print(i,'%.3f'%r,' '.join(f'{names[k]}:{v:.2f}' for k,v in zip(sel,c)),flush=True)
