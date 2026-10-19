import numpy as np, itertools, json
x=np.load('x1.npy')[:20000]; h=np.load('h1.npy')[:20000]
M=np.maximum.accumulate(x,1); m=M[:,9]; am=np.argmax(x,1); T=m[:,None]-x; I=np.eye(10)[am]
G9=np.maximum(x[:,9]-M[:,8],0); MI=m[:,None]*I
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
def partitions(w,k):
    o=np.argsort(w); out=[]
    for cuts in itertools.combinations(range(1,10),k-1):
        S=np.split(o,cuts); out.append([sorted(g.tolist()) for g in S])
        # variants dropping one group (treated as zero weight)
        if k>=1:
            for d in range(len(S)):
                out.append([sorted(g.tolist()) for t,g in enumerate(S) if t!=d])
    return out
res={}
for i in range(16):
    y=h[:,i]; r,c=fit(np.c_[T,G9,MI],y); wT=c[:10]; wA=c[11:]
    best=(-1,)
    for g in [0,1]:
        for nT in range(0,5):
            for nA in range(0,4):
                if g+nT+nA>5 or g+nT+nA==0: continue
                PT=partitions(wT,nT) if nT else [[]]
                PA=partitions(wA,nA) if nA else [[]]
                PT=[p for p in PT if len(p)>0] if nT else PT
                for pt in PT:
                    ft=[T[:,s].sum(1) for s in pt]
                    for pa in PA:
                        fa=[MI[:,s].sum(1) for s in pa]
                        cols=ft+fa+([G9] if g else [])
                        if not cols: continue
                        rr,cc=fit(np.stack(cols,1),y)
                        if rr>best[0]: best=(rr,pt,pa,g,cc.round(3).tolist())
    res[i]=best; print(i,'full %.3f'%r,'best5 %.3f'%best[0],'Tsets',best[1],'amsets',best[2],'G9',best[3],best[4],flush=True)
json.dump({k:[v[0],v[1],v[2],v[3],v[4]] for k,v in res.items()},open('s19.json','w'))
