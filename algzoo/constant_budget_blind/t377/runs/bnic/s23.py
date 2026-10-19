import numpy as np, itertools, json
x=np.load('x1.npy'); h=np.load('h1.npy')
srt=-np.sort(-x,1); am=np.argmax(x,1); m=srt[:,0]; g2=m-srt[:,1]
M=np.maximum.accumulate(x,1); T=m[:,None]-x; G9=np.maximum(x[:,9]-M[:,8],0)
S8=-np.sort(-x[:,:9],1)[:,1]; Q9=np.maximum(S8-x[:,9],0)
I=np.eye(10)[am]
X=np.c_[T,m[:,None]*I,g2[:,None]*I,G9,Q9]   # 0-9 T, 10-19 MI, 20-29 g2I, 30 G9, 31 Q9
G=X.T@X
def partitions(w,k):
    o=np.argsort(w); out=[]
    for cuts in itertools.combinations(range(1,10),k-1):
        S=np.split(o,cuts); out.append([g.tolist() for g in S])
        for d in range(len(S)):
            if len(S)>1: out.append([g.tolist() for t,g in enumerate(S) if t!=d])
    return out
res={}
for i in range(16):
    y=h[:,i]; b=X.T@y; yy=y@y; vy=y.var()*len(y)
    def r2(W):
        Gw=W.T@G@W; bw=W.T@b
        try: c=np.linalg.solve(Gw,bw)
        except np.linalg.LinAlgError: return -1,None
        sse=yy-2*c@bw+c@Gw@c; return 1-sse/vy,c
    cfull=np.linalg.lstsq(X,y,rcond=None)[0]; rfull=r2(np.eye(32))[0]
    best=(-1,)
    for extras in [(),(30,),(31,),(30,31)]:
        for nT in range(0,5):
            for nA in range(0,3):
                for nB in range(0,3):
                    if nT+nA+nB+len(extras)>5 or nT+nA+nB+len(extras)==0: continue
                    PT=partitions(cfull[0:10],nT) if nT else [[]]
                    PA=partitions(cfull[10:20],nA) if nA else [[]]
                    PB=partitions(cfull[20:30],nB) if nB else [[]]
                    for pt in PT:
                        for pa in PA:
                            for pb in PB:
                                cols=[]
                                for off,P in [(0,pt),(10,pa),(20,pb)]:
                                    for S in P:
                                        v=np.zeros(32); v[[off+j for j in S]]=1; cols.append(v)
                                for e in extras:
                                    v=np.zeros(32); v[e]=1; cols.append(v)
                                if not cols or len(cols)>5: continue
                                r,c=r2(np.stack(cols,1))
                                if r>best[0]: best=(r,pt,pa,pb,extras,c.round(3).tolist())
    res[i]=[best[0],best[1],best[2],best[3],list(best[4]),best[5]]
    print(i,'full %.3f best5 %.3f'%(rfull,best[0]),'T',best[1],'Mam',best[2],'g2am',best[3],'extras(30=G9,31=Q9)',best[4],best[5],flush=True)
json.dump(res,open('s23.json','w'))
