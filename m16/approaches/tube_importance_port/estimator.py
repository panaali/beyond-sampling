"""Per-call deterministic importance fitting with full cold numerical proxy ledger."""
import hashlib,math,time,types
from pathlib import Path
import numpy as np
from ..feynman_incremental.radial import walk_mass
from ..feynman_incremental.adapter import CORE,WalkFailure
SOURCE=Path('/workspace/clean-neurons/m16/approaches/tube/tube.py')
EXPECTED='09fc08e0e3386b664971f4cbf003824c2ff44b11fe85f7cda1b7f8f7cc60e718'
TABLES={'23':'fb01ea23e82324990c7e94ff3f354b406bc463a42f13801e94ca2d0795c7c4c6','12':'6ae92ae1e7fc5709c8f5cd13233872ded07a570657bad079320bd4ce5ee1b5c9'}

def digest(array):return hashlib.sha256(np.asarray(array,dtype='<f8').tobytes()).hexdigest()

def fit_proposal(values,costs,groups,K,score_kind="sd"):
    if score_kind not in ("sd","rms"):raise ValueError("Unknown pilot score")
    values=np.asarray(values,float);costs=np.asarray(costs,float);groups=np.asarray(groups)
    if values.shape!=costs.shape or groups.shape!=values.shape or values.ndim!=1 or not np.isfinite(values).all() or not np.isfinite(costs).all() or np.any(costs<=0) or np.any(groups!=groups.astype(int)) or np.any((groups<0)|(groups>=K)):raise ValueError('Invalid pilot data')
    groups=groups.astype(int);counts=np.bincount(groups,minlength=K)
    if np.min(counts)<2:raise ValueError('Pilot did not cover every stratum twice')
    scores=np.array([(np.std(values[groups==j],ddof=1) if score_kind=="sd" else math.sqrt(float(np.mean(values[groups==j]**2))))/math.sqrt(float(np.mean(costs[groups==j]))) for j in range(K)])
    total=float(scores.sum());q=np.full(K,1/K) if total==0 else .2/K+.8*scores/total
    if not np.isfinite(q).all() or np.min(q)<.2/K or abs(float(q.sum())-1)>1e-12:raise ArithmeticError('Invalid learned proposal')
    return q,dict(score_kind=score_kind,min_count=int(counts.min()),max_count=int(counts.max()),score_sum=total,uniform_fallback=total==0,proposal_sha256=digest(q),proposal=q.tolist(),fit_proxy=25*len(values)+200*K)

def estimate(model,budget,mode='cold',score_kind='sd'):
    start=time.process_time();components={'validation':1000};records=[];tables={};kernels={};stage='validation';failed=None
    def add(name,n):components[name]=components.get(name,0)+int(n)
    def snapshot(complete):
        return dict(complete=complete,mode=mode,score_kind=score_kind,cold=True,known_flops=sum(components.values()),components=dict(components),records=list(records),proposals=dict(tables),kernels=dict(kernels),cpu_seconds=time.process_time()-start,scalar_cost_is_proxy=True,portable_bound_certified=False,opaque_kernel_costs_certified=False,imports_and_interpreter_excluded_from_scalar_proxy=True,original_advice_fitting_cost_unknown=mode=='advice',complete_model_dependent_setup=mode!='advice',source_sha256=EXPECTED,roundoff_certified=False)
    try:
        model=tuple(np.asarray(v,float) for v in model)
        if len(model)!=3:raise ValueError('Expected Wi,Wh,Wo')
        wi,wh,wo=model;H=wi.size;add('model_validation',5*sum(x.size for x in model))
        if wi.shape!=(H,) or wh.shape!=(H,H) or wo.shape!=(10,H) or H<1 or not all(np.isfinite(x).all() for x in model):raise ValueError('Expected finite bias-free T10 model')
        budget=float(budget)
        if not math.isfinite(budget) or budget<=0 or mode not in ('uniform','advice','cold') or score_kind not in ('sd','rms'):raise ValueError('Invalid budget/mode')
        sizes={'23':max(8,int(.66*budget/8)),'12':max(8,int(.28*budget/7.1))};nb=max(11,int(.06*budget))
        if max(*sizes.values(),nb)>65536:raise ValueError('Requested batch exceeds65536')
        stage='setup';text=SOURCE.read_text()
        if hashlib.sha256(text.encode()).hexdigest()!=EXPECTED:raise RuntimeError('Pinned tube source changed')
        add('cold_geometry_tables',500*(200001+20001));module=types.ModuleType('cold_importance_tube');exec(compile(text,str(SOURCE),'exec'),module.__dict__)
        original_qmc=module._qmc
        def qmc(n,d,skip=0):
            if skip:raise ValueError('Frozen rule has no skip')
            add('sobol_and_shift',100000*d+34*d*n);kernels['sobol_initializations']=kernels.get('sobol_initializations',0)+1;kernels['sobol_coordinates']=kernels.get('sobol_coordinates',0)+n*d
            return original_qmc(n,d,skip)
        module._qmc=qmc
        def evaluate(kind,n,proposal=None,pilot=False):
            nonlocal failed
            add('face_geometry',n*(1000+230*9+80));add('radius_direction',n*500);add('aggregation',1000+20*n)
            if proposal is not None:add('importance_select_and_weight',10*len(proposal)+100*n)
            shift=(module.SHIFT+math.sqrt(2)-1)%1 if pilot else module.SHIFT
            fc=getattr(module,'faces'+kind)(n,shift=shift,q=proposal);rho=np.linalg.norm(fc['x0'],axis=1)
            if np.any(rho<=0) or not np.isfinite(rho).all():raise ArithmeticError('Invalid face norm')
            a=fc['d'].argmax(1);b=fc['d'].argmin(1);tau=np.maximum(fc['hi'],0)/(rho*math.sqrt(2));tex=fc['lo']/(rho*math.sqrt(2))
            try:mass,rowcost,ledger=walk_mass(model,fc['x0']/rho[:,None],a,b,tau,tex,top=kind=='12',details=True)
            except WalkFailure as exc:failed=exc.ledger;add('radial_walker',failed['known_flops']);raise
            add('radial_walker',ledger['known_flops']);records.append(dict(kind=kind,pilot=pilot,rows=n,face_sha256=digest(fc['x0']),mass_sha256=digest(mass),walker=ledger))
            return mass,rowcost,fc
        proposals={}
        if mode=='cold':
            stage='pilot'
            for kind,n,K in (('23',2048,360),('12',256,45)):
                mass,cost,fc=evaluate(kind,n,pilot=True);add('proposal_fit',25*n+200*K)
                proposal,fit=fit_proposal(mass,cost,fc['combo'],K,score_kind=score_kind);proposals[kind]=proposal;tables[kind]=fit
        elif mode=='advice':
            for kind in ('23','12'):
                source=SOURCE.with_name('q'+kind+'.npy');add('advice_loading_validation',20000)
                if hashlib.sha256(source.read_bytes()).hexdigest()!=TABLES[kind]:raise RuntimeError('Original advice table changed')
                q=np.load(source);K=360 if kind=='23' else 45
                if q.shape!=(K,) or not np.isfinite(q).all() or np.min(q)<=0:raise ValueError('Invalid original proposal')
                proposals[kind]=q;tables[kind]=dict(file_sha256=TABLES[kind],proposal_sha256=digest(q),historical_precompute_fpe_approx=280000,exact_precompute_cost_unknown=True)
        parts={};stage='production'
        for kind,n in sizes.items():
            mass,_,fc=evaluate(kind,n,proposal=proposals.get(kind));parts['e'+kind]=float(fc['Z']*np.mean(fc['wt']*mass))
        add('blind_transform_geometry',nb*(230*10+1000));add('blind_forward_and_ranking',nb*(10*(2*H*H+2*H)+10*(2*H-1)+100+9));add('aggregation',1000+10*nb)
        xb=module.blind_points(nb);parts['blind']=float(module.P_BLIND*np.mean(module.forward_from(model,xb).argmax(1)!=np.argsort(xb,axis=1)[:,-2]));value=1-sum(parts.values())
        if not math.isfinite(value):raise ArithmeticError('Nonfinite estimate')
    except Exception as exc:
        ledger=snapshot(False);ledger.update(failure_kind='invalid_input' if stage=='validation' else 'numerical_or_implementation',failure_stage=stage,failed_walk=failed,cost_is_partial=stage!='validation',unresolved_failed_kernel_cost=failed.get('unresolved_failed_kernel_cost',True) if failed else stage!='validation');raise WalkFailure(str(exc),ledger) from exc
    ledger=snapshot(True);ledger.update(parts=parts,requested_budget=budget,actual_fpe=sum(components.values())/CORE.fpe_flops(H,10),fpe_unit=CORE.fpe_flops(H,10),production_counts=sizes,blind_points=nb)
    return value,ledger['actual_fpe'],ledger

def est(model,budget,rng):
    value,cost,_=estimate(model,budget,mode='cold');return value,cost
