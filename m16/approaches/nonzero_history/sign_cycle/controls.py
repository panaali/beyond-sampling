"""Independent sign-rank quadrature and iid-mixture identities."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from ranks import point_weights,scalar_integrals

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from affine import rank_integrals
sys.path.pop(0)

start=time.process_time();stats=defaultdict(float);out=dict(point_controls=[],interval_controls=[])
grid=np.array([0.,1e-12,.01,.2,.5,.7,.99,1-1e-12,1.])
qx,qy=np.meshgrid(grid,grid)
for m in (2,3):
    plus=point_weights(qx,qy,m,1,stats)
    minus=point_weights(qx,qy,m,-1,stats)
    average=[(a+b)/2 for a,b in zip(plus,minus)]
    first=np.where(qx>=qy,qy**m,m*(1-qy)*qy**(m-1))
    last=np.where(qx<qy,qx**m,m*(1-qx)*qx**(m-1))
    earlier=(1-first-last)/m
    error=max(float(np.max(abs(a-b))) for a,b in zip(average,(first,last,earlier,earlier)))
    out['point_controls'].append(dict(m=m,maximum_iid_mixture_error=error));assert error<1e-12
    for y,lo,hi in [(-3.,-6.,5.),(-.1,-2.,3.),(0.,-.01,.01),(2.,-5.,6.),(5.,5.,5.0001)]:
        both=[scalar_integrals(np.array([y]),np.array([lo]),np.array([hi]),m,s,stats) for s in (1,-1)]
        old=rank_integrals(np.array([y]),np.array([lo]),np.array([hi]),m,stats)
        avg=[(a+b)/2 for a,b in zip(*both)]
        error=max(abs(avg[0][0]-old[0][0]),abs(avg[1][0]-old[1][0]),abs(avg[2][0]-old[2][0]),abs(avg[3][0]-old[3][0]),abs(avg[4][0]-old[3][0]))
        row=dict(m=m,y=y,lo=lo,hi=hi,maximum_iid_mixture_error=float(error),independent=[])
        assert error<1e-12
        for sign,calculated in zip((1,-1),both):
            # Independently integrate special-prefix probability directly in
            # x and v, using adaptive scalar quadrature and the sign density.
            n=m-1;calls=[0]
            def special(x):
                L,U=sorted((x,y))
                da,db=(max(L,0.),max(U,0.)) if sign>0 else (min(L,0.),min(U,0.))
                ea,eb=(max(U,0.),np.inf) if sign>0 else (min(U,0.),0.)
                def density(v):return np.exp(-v*v/2)/np.sqrt(2*np.pi)
                d=quad(lambda v:2*density(v)*ndtr(v)**n,da,db,epsabs=1e-10)[0]
                e=quad(lambda v:2*density(v)*n*(1-ndtr(v))*ndtr(v)**(n-1),ea,eb,epsabs=1e-10)[0]
                calls[0]+=1
                return (d+e)*density(x)
            points=sorted(set([lo,hi]+[z for z in (0.,y) if lo<z<hi]))
            direct=sum(quad(special,a,b,epsabs=1e-10,epsrel=1e-10)[0] for a,b in zip(points[:-1],points[1:]))
            delta=abs(direct-calculated[3][0])
            row['independent'].append(dict(sign=sign,value=direct,error=float(delta),outer_calls=calls[0]))
            assert delta<2e-9
        out['interval_controls'].append(row)
out.update(cpu_seconds=time.process_time()-start,work=dict(stats))
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu_seconds=out['cpu_seconds'],maximum_independent_error=max(x['error'] for r in out['interval_controls'] for x in r['independent']))))
