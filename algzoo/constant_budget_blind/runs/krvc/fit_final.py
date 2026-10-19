import numpy as np, json, sys
from lib import feats
x=np.load('x1.npy')[:8000];H=np.load('H1.npy')[:8000]
F=feats(x); names=list(F); X=np.stack([F[n] for n in names],1); X=np.nan_to_num(X)
xv=np.load('x1.npy')[80000:100000];Hv=np.load('H1.npy')[80000:100000]; Fv=feats(xv)
SPEC={0:(['r(S9-x9)n8'],3),13:(['r(S9-x9)n8'],3),29:(['r(S9-x9)n8'],3),6:(['r(S9-x9)'],3),18:(['r(S9-x9)'],3),
 1:(['g12n8'],0),2:(['r(x3-to)','top2_3','am3','r(x0-mo)'],0),7:(['r(x9-M9)'],0),9:(['S9','T9','x9','g12'],0),
 12:(['r(S9-x9)'],0),15:(['r(x1-to)','top2_1','am1'],1),16:(['r(x9-S9)'],0),17:(['M9','S9'],0),
 19:(['r(x6-to)','top2_6','am6','r(x0-mo)'],0),22:(['r(x7-to)','top2_7','am7'],1),23:(['x9-M9','r(x9-S9)','am8'],1),
 27:(['am[2,6]','rmo[2,6]'],0),31:(['S9'],3),14:(['r(so-x8)'],3),28:(['A_P6'],3),8:(['A_P6'],3)}
def rf(A,y,it=12):
    m=np.ones(len(y),bool)
    for _ in range(it):
        c,*_=np.linalg.lstsq(A[m],y[m],rcond=None); m2=(y>1e-6)|(A@c>0)
        if (m2==m).all(): break
        m=m2
    return 1-((np.maximum(A@c,0)-y)**2).mean()/y.var(),c
out={}
for i in [int(a) for a in sys.argv[1:]]:
    y=H[:,i]; seed,k=SPEC.get(i,([],4)); cols=[names.index(s) for s in seed]
    for step in range(k):
        A=np.c_[X[:,cols],np.ones(len(y))]; r0,c=rf(A,y); res=y-np.maximum(A@c,0)
        Xc=X-X.mean(0); cor=np.abs(Xc.T@(res-res.mean()))/(np.sqrt((Xc**2).sum(0))+1e-9)
        cand=[j for j in np.argsort(-cor)[:50] if j not in cols]
        best=max(((rf(np.c_[X[:,cols+[j]],np.ones(len(y))],y)[0],j) for j in cand))
        cols.append(best[1])
    A=np.c_[X[:,cols],np.ones(len(y))]; r,c=rf(A,y)
    Av=np.stack([Fv[names[j]] for j in cols]+[np.ones(len(xv))],1); yv=Hv[:,i]
    rv=1-((np.maximum(Av@c,0)-yv)**2).mean()/yv.var()
    out[i]=dict(cols=[names[j] for j in cols],c=[float(v) for v in c],r2=float(rv))
    print(i,out[i],flush=True)
json.dump(out,open('fit_%s.json'%sys.argv[1],'w'))
