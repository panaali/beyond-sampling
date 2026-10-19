import numpy as np, itertools
x=np.load('x1.npy'); h=np.load('h1.npy')
srt=-np.sort(-x,1); am=np.argmax(x,1); m=srt[:,0]; g2=m-srt[:,1]
M=np.maximum.accumulate(x,1); T=m[:,None]-x; G9=np.maximum(x[:,9]-M[:,8],0)
S8=-np.sort(-x[:,:9],1)[:,1]; Q9=np.maximum(S8-x[:,9],0); I=np.eye(10)[am]
negsum=np.maximum(-x,0).sum(1)
blocks={'T':T,'Mam':m[:,None]*I,'g2am':g2[:,None]*I,'x':x,'M':M}
names=[];cols=[]
for k,B in blocks.items():
    for j in range(10): names.append(f'{k}{j}'); cols.append(B[:,j])
for k,v in [('G9',G9),('Q9',Q9),('negsum',negsum),('g2',g2),('M9amE',m*(am%2==0)),('g2amE',g2*(am%2==0))]:
    names.append(k); cols.append(v)
X=np.stack(cols,1); G=X.T@X; nb=X.shape[1]
idx={n:i for i,n in enumerate(names)}
import sys
for i in [int(a) for a in sys.argv[1:]]:
    y=h[:,i]; b=X.T@y; yy=y@y; vy=y.var()*len(y)
    cf=np.linalg.lstsq(np.c_[T,m[:,None]*I,g2[:,None]*I,G9],y,rcond=None)[0]
    pool={}; 
    for n in names: v=np.zeros(nb); v[idx[n]]=1; pool[n]=v
    for off,(blk,w) in enumerate([('T',cf[:10]),('Mam',cf[10:20]),('g2am',cf[20:30])]):
        o=np.argsort(w)
        for k in range(2,5):
            for cuts in itertools.combinations(range(1,10),k-1):
                for S in np.split(o,cuts):
                    if 1<len(S)<10:
                        v=np.zeros(nb); v[[idx[f'{blk}{j}'] for j in S]]=1; pool[blk+'{'+','.join(map(str,sorted(S)))+'}']=v
    pn=list(pool); P=np.stack([pool[n] for n in pn],1)
    PG=P.T@G@P; Pb=P.T@b
    def sc(sel):
        Gw=PG[np.ix_(sel,sel)]; bw=Pb[sel]
        if np.linalg.cond(Gw)>1e9: return -1
        try: c=np.linalg.solve(Gw,bw)
        except np.linalg.LinAlgError: return -1
        return 1-(yy-c@bw)/vy
    sel=[]
    for st in range(5): sel.append(max((k for k in range(len(pn)) if k not in sel),key=lambda k:sc(sel+[k])))
    imp=True
    while imp:
        imp=False; cur=sc(sel)
        for p in range(5):
            for k in range(len(pn)):
                if k in sel: continue
                t=sel.copy(); t[p]=k; r=sc(t)
                if r>cur+1e-4: sel=t; cur=r; imp=True
    c=np.linalg.solve(PG[np.ix_(sel,sel)],Pb[sel])
    print(i,'%.3f'%sc(sel),' '.join(f'{pn[k]}:{v:.2f}' for k,v in zip(sel,c)),flush=True)
