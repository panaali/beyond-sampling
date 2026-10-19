import numpy as np, probe, json, sys
from fitlib import pf, clus, unconstrained, r2
from scipy.optimize import least_squares
rng=np.random.default_rng(11)
x=rng.standard_normal((40000,10));h=probe.hidden(x)
xt=np.random.default_rng(12).standard_normal((40000,10));ht=probe.hidden(xt)
F=pf(x);Ft=pf(xt)
def z(form,x,F,th,cg,ch,A):
    P,Q,M,S,x9=F['P'],F['Q'],F['M'],F['S'],F['x9']
    if form=='new': return th[0]*(x9-M)
    if form=='top2x9': return th[0]*x9-th[1]*S+th[2]*M
    if form=='win2':
        w=-np.sort(-x[:,:5],1)
        return th[0]*w[:,1]+th[1]*np.maximum(x[:,2],0)+th[2]*S+th[3]*w[:,0]
    g=th[:3][cg[P]]*M
    if form=='G3S_U': return g+th[3]*S+th[4]*x9
    if form=='G3H2': return g+th[3:5][ch[Q]]*S
    if form=='G3S_mem': return g+th[3]*S+th[4]*np.maximum(x[:,A],0).sum(1)
def fitone(form,hi,cg,ch,A,th0):
    r=least_squares(lambda th:np.maximum(0,z(form,x,F,th,cg,ch,A))-hi,th0)
    return r.x, r2(np.maximum(0,z(form,x,F,r.x,cg,ch,A)),hi)
def init(form,G,H,cg,ch):
    if form in('new',):return np.r_[20.]
    if form=='top2x9':return np.r_[8.,6,0]
    if form=='win2':return np.r_[8.,5,-2,0]
    t=[G[cg==k].mean() if (cg==k).any() else 0 for k in range(3)]
    if form=='G3H2':return np.r_[t,[H[ch==k].mean() if (ch==k).any() else 0 for k in range(2)]]
    return np.r_[t,0,0]
def search(i,form,A=[]):
    hi=h[:,i];G,H,u=unconstrained(F,hi);cg=clus(G,3);ch=clus(H,2)
    th,sc=fitone(form,hi,cg,ch,A,init(form,G,H,cg,ch))
    if form.startswith('G3'):
        improved=True
        while improved:
            improved=False
            for p in range(9):
                for k in range(3):
                    if k==cg[p]:continue
                    c2=cg.copy();c2[p]=k
                    t2,s2=fitone(form,hi,c2,ch,A,th)
                    if s2>sc+1e-3: cg,th,sc,improved=c2,t2,s2,True
            if form=='G3H2':
                for q in range(9):
                    c2=ch.copy();c2[q]=1-c2[q]
                    t2,s2=fitone(form,hi,cg,c2,A,th)
                    if s2>sc+1e-3: ch,th,sc,improved=c2,t2,s2,True
    te=r2(np.maximum(0,z(form,xt,Ft,th,cg,ch,A)),ht[:,i])
    return dict(i=i,form=form,th=list(map(float,th)),cg=list(map(int,cg)),ch=list(map(int,ch)),A=list(map(int,A)),train=float(sc),test=float(te))
plan=json.loads(sys.argv[1])
out=[]
for i,specs in plan.items():
    best=None
    for sp in specs:
        res=search(int(i),sp[0],sp[1] if len(sp)>1 else [])
        print(i,res['form'],res['A'],'train %.3f test %.3f'%(res['train'],res['test']),np.round(res['th'],2),res['cg'],res['ch'],flush=True)
        if best is None or res['test']>best['test']: best=res
    out.append(best)
json.dump(out,open(sys.argv[2],'w'))
