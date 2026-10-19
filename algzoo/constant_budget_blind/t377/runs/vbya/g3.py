import numpy as np, probe, json
np.set_printoptions(precision=3,suppress=True,linewidth=220)
rng=np.random.default_rng(8)
x=rng.standard_normal((30000,10)); H=probe.hidden(x)
xt=rng.standard_normal((30000,10)); Ht=probe.hidden(xt)
R=lambda z:np.maximum(z,0)
def basics(x):
    pre=x[:,:9]; ps=-np.sort(-pre,1); pam=pre.argmax(1)
    return ps[:,0],ps[:,1],pam
def feat(x,T):
    p1,p2,pam=basics(x); D=np.isin(pam,T)*(p1-p2)
    return np.c_[p1,p2,D]
def fitg(g,x,T):
    A=np.c_[feat(x,T),np.ones(len(g))]; c,*_=np.linalg.lstsq(A,g,rcond=None); r=g-A@c; return c,1-r.var()/g.var()
out={}
for sgn,idx in [(-1,[3,7,9,17,18,20,22,28,30]),(1,[8,13,19])]:
    y=x.copy(); y[:,9]=20*sgn; ha=probe.hidden(y); y[:,9]=25*sgn; hb=probe.hidden(y)
    for i in idx:
        w=np.median((hb[:,i]-ha[:,i])/(5*sgn)); g=ha[:,i]-w*20*sgn
        T=[]; best=fitg(g,x,T)[1]
        imp=True
        while imp:
            imp=False
            for t in range(9):
                T2=[u for u in T if u!=t] if t in T else sorted(T+[t])
                r=fitg(g,x,T2)[1]
                if r>best+1e-4: best=r; T=T2; imp=True
        c,r=fitg(g,x,T)
        hp=R(w*xt[:,9]+np.c_[feat(xt,T),np.ones(len(xt))]@c)
        hr=1-((Ht[:,i]-hp)**2).mean()/Ht[:,i].var()
        print(i,'w %.3f T=%s coef(ps1,ps2,D,1)'%(w,T),c,'gR2 %.3f  hR2(test) %.3f'%(r,hr))
        out[i]=dict(w=float(w),T=T,c=[float(v) for v in c],hR2=float(hr))
json.dump(out,open('fit_x9.json','w'))
