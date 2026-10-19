import numpy as np,sys,pickle
from core import feat
from relufit import relufit
d=np.load('d1.npz'); x,h=d['x'][:15000],d['h'][:15000]
p=x[:,:9]; Q=-np.sort(-p,1); s=np.argsort(-p,1)
cols=[]
for j in range(9):
    for a,b in [(0,0),(0,1),(1,0),(1,1)]: cols.append((s[:,a]==j)*Q[:,b])
T=np.c_[np.stack(cols,1),Q[:,2],x[:,9]]
base=['Q1','Q2','Q3','x9','rnew8','gap']+[f'x|{j}' for j in range(9)]+[f'rxQ3|{j}' for j in range(9)]+[f'rxQ2|{j}' for j in range(9)]+[f'rQ2x|{j}' for j in range(9)]+[f'rQ1x|{j}' for j in range(9)]+['rx8S7','rS7x8','rP7x8']
def fsel(A,y,ch,k):
    m=y>1e-6
    X=A[:,ch+[k]]
    for it in range(5):
        co=np.linalg.lstsq(X[m],y[m],rcond=None)[0]; m=(y>1e-6)|(X@co>0)
    return 1-((np.maximum(X@co,0)-y)**2).mean()/y.var(),co
res={}
for i in map(int,sys.argv[1:]):
    y=h[:,i]
    co,_=relufit(T,y); C=co[:36].reshape(9,4)
    sets=set()
    for kind,rows in [('am',(0,1)),('as',(2,3))]:
        for prof in [C[:,rows[0]],C[:,rows[1]],C[:,rows[0]]-C[:,rows[1]]]:
            o=np.argsort(prof)
            for kk in range(1,5):
                sets.add((kind,''.join(map(str,sorted(o[-kk:])))))
                sets.add((kind,''.join(map(str,sorted(o[:kk])))))
    names=list(base)
    for kind,S in sets:
        for v in (['Q1','Q2','g12'] if kind=='am' else ['Q1','Q2','g23']): names.append(f'{kind}|{v}|{S}')
    A=np.stack([feat(n,x) for n in names],1)
    ch=[]; log=[]; best=None
    for st in range(5):
        best=None
        for k in range(len(names)):
            if k in ch: continue
            R,c=fsel(A,y,ch,k)
            if best is None or R>best[0]: best=(R,k,c)
        ch.append(best[1]); log.append((names[best[1]],round(best[0],4),best[2].copy()))
    res[i]=log
    print(i,' '.join(f'{n}:{R}' for n,R,_ in log),flush=True)
pickle.dump(res,open('search3_%s.pkl'%sys.argv[1],'wb'))
