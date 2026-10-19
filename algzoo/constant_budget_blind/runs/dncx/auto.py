import numpy as np, json, sys
from featlib import base, feat
from lib2 import fitrelu
d=np.load('d1.npz'); x=d['x'][:12000].astype(float); h=d['h'][:12000]
b=base(x)
P=np.c_[[b[k]*(b['p'+k]==j) for k in 'ABC' for j in range(9)]+[x[:,9]]].T
gen=['A','B','C','x9','Ap','Bp','Cp','Aq','Bq','sum9','sum10']+['x%d'%j for j in range(9)]+['m%d'%t for t in range(1,9)]+['m2_%d'%t for t in range(1,9)]+\
    ['xjB+%d'%j for j in range(9)]+['Bxj+%d'%j for j in range(9)]+['Cxj+%d'%j for j in range(9)]+['gap%d'%t for t in range(1,10)]
out={}
for i in map(int,sys.argv[1:]):
    y=h[:,i]; r2p,c=fitrelu(P,y,6)
    cand=list(gen)
    for ki,k in enumerate('ABC'):
        o=np.argsort(c[9*ki:9*ki+9])
        for kk in range(1,5):
            cand.append(k+'@'+','.join(map(str,sorted(o[-kk:])))); cand.append(k+'@'+','.join(map(str,sorted(o[:kk]))))
    cand=list(dict.fromkeys(cand)); Fm={s:feat(b,s) for s in cand}
    sel=[]; hist=[]
    for step in range(5):
        best=None
        for q in cand:
            if q in sel: continue
            r2,cc=fitrelu(np.c_[[Fm[z] for z in sel+[q]]].T,y,3)
            if best is None or r2>best[0]: best=(r2,q)
        sel.append(best[1]); hist.append(round(best[0],3))
    # swap refinement
    for _ in range(2):
        for pos in range(5):
            cur=fitrelu(np.c_[[Fm[z] for z in sel]].T,y,3)[0]
            for q in cand:
                if q in sel: continue
                t=sel[:pos]+[q]+sel[pos+1:]; r2=fitrelu(np.c_[[Fm[z] for z in t]].T,y,3)[0]
                if r2>cur+1e-4: sel=t; cur=r2
    r2,cc=fitrelu(np.c_[[Fm[z] for z in sel]].T,y,10)
    print(i,'prof %.3f'%r2p,'hist',hist,'final %.3f'%r2,list(zip(sel,np.round(cc,2))),flush=True)
    out[i]=dict(specs=sel,coefs=cc.tolist())
json.dump(out,open('auto_%s.json'%sys.argv[1],'w'))
