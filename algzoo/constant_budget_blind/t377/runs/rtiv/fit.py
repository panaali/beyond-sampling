import numpy as np, probe, re, importlib
from scipy.optimize import least_squares
import labels as L
rng=np.random.default_rng(2);x=rng.standard_normal((100000,10));H=probe.hidden(x)
rng=np.random.default_rng(3);xt=rng.standard_normal((100000,10));Ht=probe.hidden(xt)
F=L._feat(x);Ft=L._feat(xt)
plan={0:[('G4',{'P':[0,4,5,7]})],1:[('G3',{'P':[9]})],2:[('G1',{'P':[0,7,8]})],
4:[('G3',{'P':[1,5,7,8]})],7:[('G3',{'P':[0,2,9]})],9:[('G3',{'P':[6]}),('Hs',{'Q':[0,2,7,8],'P':[6]})],
10:[('G4',{'P':[3,7,8]})],11:[('G1',{'P':[0,5,7,8]})],13:[('G3',{'P':[1,3,9]}),('Hs',{'Q':[1,3,4,5,6,9],'P':[1,3,9]})],15:[('G1',{'P':[4,5,8]}),('H78',{'Q':[5,9],'P':[4,5,8]})],
3:[('gap',{'t':9})],6:[('gap',{'t':9})],12:[('gap',{'t':9})],8:[('gap',{'t':7})],14:[('gap14',{})],5:[('up',{})]}
nparam={'gap':5,'gap14':4,'Hs':5,'H78':5,'up':2,'G1':5,'G3':5,'G4':5}
out={}
for i in range(16):
    best=None
    for form,st in plan[i]:
        k=nparam[form];y=H[:,i]
        res=lambda c: L._eval(form,st,c,F)-y
        c0=np.ones(k)*3.0
        if form.startswith('G'):
            J=np.stack([L._eval(form,st,np.eye(k)[j],F) for j in range(k)],1);c,*_=np.linalg.lstsq(J,y,rcond=None)
        else:
            if form=='up': c0=np.array([9.,9.])
            if form=='gap14': c0=np.array([7.,3.,6.,2.5])
            if form=='gap': c0=np.array([12,0,6,6,12.])
            c=least_squares(res,c0).x
        p=L._eval(form,st,c,Ft);r=1-((Ht[:,i]-p)**2).mean()/Ht[:,i].var()
        print(i,form,st,"test R2 %.4f"%r,np.round(c,3))
        if best is None or r>best[0]: best=(r,form,st,c)
    out[i]=best
txt="\n".join(f"    {i}: ({out[i][1]!r}, {out[i][2]!r}, [{', '.join('%.4f'%v for v in out[i][3])}]),  # test R2 {out[i][0]:.3f}" for i in range(16))
src=open('labels.py').read()
src=re.sub(r"#SPEC_START.*#SPEC_END","#SPEC_START\n"+txt+"\n#SPEC_END",src,flags=re.S)
open('labels.py','w').write(src)
