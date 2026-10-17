"""Counted adapter from Feynman excluded windows to the audited v2 walker."""
import hashlib,importlib.util
from pathlib import Path
import numpy as np

SOURCE=Path('/workspace/clean-neurons/m16/approaches/sci_tao_v2/core.py')
EXPECTED='aea5f9008fe9cc8a5436226f1057b5e0527f119d832df536127d8370bc7968f9'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=EXPECTED:
    raise RuntimeError('Shared walker revision needs review')
spec=importlib.util.spec_from_file_location('_feynman_port_incremental',SOURCE)
CORE=importlib.util.module_from_spec(spec);spec.loader.exec_module(CORE)

class WalkFailure(RuntimeError):
    def __init__(self,message,ledger):
        super().__init__(message);self.ledger=ledger

def walk_mass(model,y,a,b,tau,tex,top=False,eps=1e-10,max_iter=2000,details=False):
    model=tuple(np.asarray(x,dtype=float) for x in model)
    y=np.asarray(y,dtype=float);a=np.asarray(a);b=np.asarray(b)
    tau=np.asarray(tau,dtype=float);tex=np.asarray(tex,dtype=float)
    if len(model)!=3 or y.ndim!=2:raise ValueError('Expected model triple and matrix of face points')
    wi,wh,wo=model;n,T=y.shape;H=len(wi)
    local=100+5*(n*T+H+H*H+T*H)+100*n
    ledger=dict(shared_source_sha256=EXPECTED,adapter_flops=local,calls=[],
                known_flops=local,complete=False,cost_is_partial=False)
    valid=(wi.shape==(H,) and wh.shape==(H,H) and wo.shape==(T,H) and
           all(x.shape==(n,) for x in (a,b,tau,tex)) and T>=2)
    if not valid:raise WalkFailure('Invalid model/window shapes',ledger)
    if not (all(np.isfinite(x).all() for x in (*model,y,tau,tex)) and
            np.all(tau>=0) and np.all(tex>=0) and
            np.all(a==a.astype(int)) and np.all(b==b.astype(int)) and
            np.all((a>=0)&(a<T)&(b>=0)&(b<T)&(a!=b)) and
            isinstance(max_iter,(int,np.integer)) and max_iter>=0 and
            np.isfinite(eps) and eps>=0):
        raise WalkFailure('Invalid finite model/window/index/cap input',ledger)
    a=a.astype(int);b=b.astype(int);unit=CORE.fpe_flops(H,T)
    mass=np.zeros(n);cost=np.full(n,local/max(1,n)/unit)
    kind='12' if top else '23';active=np.flatnonzero(tau>tex)

    def evaluate(ids,width,phase):
        try:
            result=CORE.walk_mass(model,y[ids],a[ids],b[ids],width,kind,
                                  eps=eps,max_cells=max_iter,details=True)
        except Exception as exc:
            work=None;trace=exc.__traceback__
            while trace is not None:
                if trace.tb_frame.f_code.co_filename==str(SOURCE):
                    work=trace.tb_frame.f_locals.get('work',work)
                trace=trace.tb_next
            known=sum(int(v.sum()) for k,v in (work or {}).items() if k.endswith('_flops'))
            ledger['known_flops']+=known
            ledger.update(failure_phase=phase,cost_is_partial=True,
                          failed_call_known_flops=known,
                          failed_call_counters={k:int(v.sum()) for k,v in (work or {}).items()},
                          unresolved_failed_kernel_cost=not(isinstance(exc,RuntimeError) and 'max_cells=' in str(exc)))
            raise WalkFailure(str(exc),ledger) from exc
        value,fpe,cells,work=result
        flops=sum(int(v.sum()) for k,v in work.items() if k.endswith('_flops'))
        ledger['known_flops']+=flops
        ledger['calls'].append(dict(phase=phase,windows=len(ids),flops=flops,
            counters={k:int(v.sum()) for k,v in work.items()}))
        cost[ids]+=fpe
        return value

    if len(active):mass[active]=evaluate(active,tau[active],'full')
    inner=active[tex[active]>0]
    if len(inner):mass[inner]-=evaluate(inner,tex[inner],'excluded')
    combine=10*n+20;ledger['known_flops']+=combine;ledger['adapter_flops']+=combine
    if n:cost+=combine/n/unit
    if np.any(mass<0) or not np.isfinite(mass).all():
        ledger['cost_is_partial']=False
        raise WalkFailure('Window subtraction produced negative/nonfinite mass',ledger)
    ledger.update(complete=True,fpe=float(ledger['known_flops']/unit),
                  empty_windows=n-len(active),shared_calls=len(ledger['calls']))
    if n:assert abs(float(cost.sum())-ledger['fpe'])<=1e-10*max(1.,ledger['fpe'])
    result=(mass,cost)
    return result+(ledger,) if details else result
