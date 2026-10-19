import numpy as np, feats, sys
n=int(sys.argv[1]) if len(sys.argv)>1 else 5
neur=[int(a) for a in sys.argv[2:]] if len(sys.argv)>2 else range(16)
x=np.load('x1.npy')[:30000]; h=np.load('h1.npy')[:30000]
F=feats.pool3(x); names=list(F); A=np.stack([F[k] for k in names],1)
def score(sel,y):
    B=A[:,sel]; c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
for i in neur:
    y=h[:,i]; sel=[]
    for step in range(n):
        sel.append(max((k for k in range(A.shape[1]) if k not in sel),key=lambda k:score(sel+[k],y)[0]))
    improved=True
    while improved:
        improved=False; cur=score(sel,y)[0]
        for p in range(n):
            for k in range(A.shape[1]):
                if k in sel: continue
                t=sel.copy(); t[p]=k; r=score(t,y)[0]
                if r>cur+1e-4: sel=t; cur=r; improved=True
    r,c=score(sel,y)
    print(i,'%.3f'%r,' '.join(f'{names[k]}:{v:.2f}' for k,v in zip(sel,c)),flush=True)
