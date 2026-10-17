"""Independent positive radial integrals and exact-rational outcome intervals."""
import json,time,sys
from pathlib import Path
from fractions import Fraction
import numpy as np
import mpmath as mp
from .radial import radial_interval,walk_mass,A
from .adapter import WalkFailure
from .check_adapter import load,REVIEW
HERE=Path(__file__).parent

def main():
    start=time.process_time();mp.mp.dps=70
    amplitude=mp.sqrt(2)*(mp.sqrt(2)*24/mp.mpf('11.631728396567448'))
    def real(s):
        f=Fraction(s);return mp.mpf(f.numerator)/f.denominator
    def integral(a,b):
        a,b=real(a),real(b);d=b-a
        cuts=[mp.mpf(0)]+[(mp.mpf(x)-a)/d for x in (-10,-1,0,1,10) if a<x<b]+[mp.mpf(1)] if d>0 else []
        return amplitude*d*mp.quad(lambda u:(1+(a+d*u)**2)**-5,cuts) if d>0 else mp.mpf(0)
    out=dict(scalar=[],fixtures=[],caps=[],random_model_inputs=0)
    pairs=[(-2.,3.),(-.1,.2),(0.,1.),(-100.,100.)]
    for a in (0.,1.,8.,20.):
        for b in (np.nextafter(a,np.inf),a+.1,a+1):pairs.extend(((a,b),(-b,-a)))
    for a,b in pairs:
        v,bd,use=radial_interval(np.array([a]),np.array([b]));ref=integral(a,b)
        rel=float(abs(mp.mpf(float(v[0]))-ref)/ref) if ref else 0.
        row=dict(lo=a,hi=b,mass=float(v[0]),reference=str(ref),relative_error=rel,midpoint=bool(use[0]),bound=float(bd[0]))
        out['scalar'].append(row)
        if b-a>1e-320:assert rel<1e-11,(a,b,rel)
    # Full integral checked independently, not by assuming normalization.
    full=2*amplitude*mp.quad(lambda t:(1+t*t)**-5,[0,1,mp.inf])
    out['full_line_mass']=str(full);out['full_line_minus_2sqrtpi']=float(full-2*mp.sqrt(mp.pi))
    assert abs(out['full_line_minus_2sqrtpi'])<1e-14
    reference=load('radial_exact_reference',REVIEW/'rational_line.py');sys.modules['rational_line']=reference
    fixtures=load('radial_independent_fixtures',REVIEW/'check.py').fixtures()
    for name,model,y,a,b in fixtures:
        direction=np.zeros(len(y));direction[a]=1/np.sqrt(2);direction[b]=-1/np.sqrt(2)
        for top in (False,True):
            for exclusion in (0.,.0625,np.nextafter(.25,0.)):
                if time.process_time()-start>18:raise TimeoutError('Radial20CPU guard')
                neg,pos=(a,b) if top else (b,a);truth=mp.mpf(0)
                for lo,hi in ((-.25,-exclusion),(exclusion,.25)):
                    regions=reference.wrong_mass(model,y,direction,lo,hi,neg,pos,deadline=start+18)['intervals']
                    truth+=sum((integral(r['lo'],r['hi']) for r in regions if r['winner']!=r['target']),mp.mpf(0))
                mass,cost,ledger=walk_mass(model,y[None],np.array([a]),np.array([b]),np.array([.25]),np.array([exclusion]),top=top,details=True)
                error=float(abs(mp.mpf(float(mass[0]))-truth));assert error<1e-12
                out['fixtures'].append(dict(name=name,top=top,exclusion=float(exclusion),mass=float(mass[0]),reference=str(truth),absolute_error=error,known_flops=ledger['known_flops'],counters=ledger['native_interval_counters']))
    name,model,y,a,b=fixtures[4]
    for cap in (0,1):
        try:walk_mass(model,y[None],np.array([a]),np.array([b]),np.array([.25]),np.zeros(1),max_iter=cap)
        except WalkFailure as exc:out['caps'].append(dict(cap=cap,failure=str(exc),ledger=exc.ledger))
        else:raise AssertionError('Cap returned completed integral')
    out.update(complete=True,cpu_seconds=time.process_time()-start,amplitude=A)
    (HERE/'radial_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(cpu_seconds=out['cpu_seconds'],fixtures=len(out['fixtures']),max_error=max(r['absolute_error'] for r in out['fixtures']))))
if __name__=='__main__':main()
