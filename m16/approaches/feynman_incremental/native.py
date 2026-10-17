"""Local interval-only adaptation; gate/cache algorithm is pinned shared v2."""
import inspect,math
import numpy as np
from scipy.special import ndtr
from .adapter import CORE,SOURCE,EXPECTED,WalkFailure,walk_mass as validate_empty

def normal_interval(lo,hi):
    """Positive interval mass; only midpoint quadrature remainder is bounded."""
    lo=np.asarray(lo,dtype=float);hi=np.asarray(hi,dtype=float)
    delta=np.maximum(hi-lo,0.)
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        midpoint=delta/math.sqrt(2*math.pi)*np.exp(-.5*lo*lo-.5*lo*delta-.125*delta*delta)
        near=np.where(lo>0,lo,np.where(hi<0,hi,0.))
        local=(np.maximum(lo*lo,hi*hi)+1)*np.exp(-.5*near*near)/math.sqrt(2*math.pi)
        bound=delta*delta*delta*np.minimum(1/math.sqrt(2*math.pi),local)/24
        use=(delta>0)&np.isfinite(midpoint)&(bound<=np.finfo(float).eps*midpoint)
    positive=lo>=0
    lower=np.where(positive,-hi,lo);upper=np.where(positive,-lo,hi)
    plain=ndtr(upper)-ndtr(lower)
    value=np.where(delta>0,np.where(use,midpoint,plain),0.)
    return value,np.where(use,bound,0.),use

def walk_mass(model,y,a,b,tau,tex,top=False,eps=1e-10,max_iter=2000,details=False):
    y=np.asarray(y,dtype=float);tau=np.asarray(tau,dtype=float);tex=np.asarray(tex,dtype=float)
    a=np.asarray(a);b=np.asarray(b)
    if not (np.isfinite(tex).all() and np.all(tex>=0) and np.isfinite(a).all() and np.isfinite(b).all()):
        raise WalkFailure('Invalid exclusion/index input',dict(known_flops=100,complete=False,cost_is_partial=False))
    # Exercise the existing validation path with explicitly empty intervals.
    _,cost,ledger=validate_empty(model,y,a,b,tau,np.maximum(tex,tau),top,eps,max_iter,details=True)
    ledger.update(complete=False,method='interval_native',calls=[])
    model=tuple(np.asarray(v,dtype=float) for v in model);n,T=y.shape;H=len(model[0]);unit=CORE.fpe_flops(H,T)
    setup=500+40*n;ledger['known_flops']+=setup;ledger['adapter_flops']+=setup
    if n:cost+=setup/n/unit
    mass=np.zeros(n);active=np.flatnonzero(tau>tex)
    per_row=np.zeros(len(active),dtype=np.int64)
    counters=dict(normal_intervals=0,midpoint_intervals=0,cdf_evaluations=0,
                  exp_evaluations=0,midpoint_quadrature_bound=0.)

    def interval(lo,hi):
        value,bound,use=normal_interval(lo,hi)
        counters['normal_intervals']+=len(lo);counters['midpoint_intervals']+=int(use.sum())
        counters['cdf_evaluations']+=2*len(lo);counters['exp_evaluations']+=2*len(lo)
        return value,np.where(use,bound,0.)

    def wrong(ids,lo,hi,clo,chi,bad):
        full,ef=interval(lo,hi)
        left,el=interval(lo,np.clip(clo,lo,hi))
        right,er=interval(np.clip(chi,lo,hi),hi)
        correct=(chi>clo)&~bad
        result=np.where(correct,left+right,full)
        counters['midpoint_quadrature_bound']+=float(np.where(correct,el+er,ef).sum())
        per_row[ids]+=940  # 3*300 interval allowance +40 assembly.
        return result

    text=inspect.getsource(CORE.walk_mass)
    replacements={
        'details=False):':'details=False, exclude=None):',
        '            valid = p_hi > p_lo\n':
        '            if side == 0:\n                p_hi = np.minimum(p_hi, -exclude[active])\n            else:\n                p_lo = np.maximum(p_lo, exclude[active])\n            valid = p_hi > p_lo\n',
        '            correct = np.where((chi > clo) & ~bad.any(axis=1), ndtr(chi) - ndtr(clo), 0.)\n            mass[ids] += ndtr(p_hi[valid]) - ndtr(p_lo[valid]) - correct\n':
        '            mass[ids] += _wrong_interval_mass(ids, p_lo[valid], p_hi[valid], clo, chi, bad.any(axis=1))\n'}
    for old,new in replacements.items():
        if text.count(old)!=1:raise RuntimeError('Unexpected pinned interval source layout')
        text=text.replace(old,new)
    namespace=dict(CORE.__dict__);namespace['_wrong_interval_mass']=wrong
    exec(compile(text,__file__,'exec'),namespace)
    if len(active):
        try:
            values,fpe,cells,work=namespace['walk_mass'](model,y[active],a[active],b[active],tau[active],
                '12' if top else '23',eps=eps,max_cells=max_iter,details=True,exclude=tex[active])
        except Exception as exc:
            work={};trace=exc.__traceback__
            while trace:
                if trace.tb_frame.f_code.co_name=='walk_mass' and 'work' in trace.tb_frame.f_locals:
                    work=trace.tb_frame.f_locals['work']
                trace=trace.tb_next
            shared=sum(int(v.sum()) for k,v in work.items() if k.endswith('_flops'))
            ledger.update(known_flops=ledger['known_flops']+shared+int(per_row.sum()),
                cost_is_partial=True,failed_call_known_flops=shared,
                unresolved_failed_kernel_cost=not(isinstance(exc,RuntimeError) and 'max_cells=' in str(exc)),
                native_interval_counters=counters)
            raise WalkFailure(str(exc),ledger) from exc
        shared=sum(int(v.sum()) for k,v in work.items() if k.endswith('_flops'))
        mass[active]=values;cost[active]+=fpe+per_row/unit
        ledger['known_flops']+=shared+int(per_row.sum())
        ledger['calls'].append(dict(phase='native_annulus',windows=len(active),flops=shared,
            counters={k:int(v.sum()) for k,v in work.items()}))
    ledger.update(complete=True,shared_calls=len(ledger['calls']),native_interval_counters=counters,
        native_interval_flops=int(per_row.sum()),fpe=ledger['known_flops']/unit,
        roundoff_certified=False)
    if np.any(mass<0) or not np.isfinite(mass).all():
        ledger['complete']=False
        raise WalkFailure('Invalid native interval mass',ledger)
    if n:assert abs(float(cost.sum())-ledger['fpe'])<1e-10*max(1.,ledger['fpe'])
    return (mass,cost,ledger) if details else (mass,cost)
