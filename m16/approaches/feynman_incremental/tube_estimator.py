"""Uniform-face radial tube port; explicit cold numerical cost proxy."""
import hashlib,math,time,types
from pathlib import Path
import numpy as np
from .radial import walk_mass
from .adapter import CORE,WalkFailure
SOURCE=Path('/workspace/clean-neurons/m16/approaches/tube/tube.py')
EXPECTED='09fc08e0e3386b664971f4cbf003824c2ff44b11fe85f7cda1b7f8f7cc60e718'

def estimate(model,budget,legacy=False):
    start=time.process_time();costs={'validation':1000};kernels={};records=[];stage='validation'
    digest=hashlib.sha256();face_digest=hashlib.sha256();failed_walk=None
    def add(k,n):costs[k]=costs.get(k,0)+int(n)
    def snapshot(complete):
        return dict(complete=complete,known_flops=sum(costs.values()),components=dict(costs),kernels=dict(kernels),
            records=records,node_sha256=digest.hexdigest(),face_sha256=face_digest.hexdigest(),
            cpu_seconds=time.process_time()-start,source_sha256=EXPECTED,uniform_faces=True,
            scalar_cost_is_proxy=True,portable_bound_certified=False,opaque_kernel_costs_certified=False,
            legacy_walker_cost_missing_from_proxy=legacy,cold=True,roundoff_certified=False,
            imports_and_interpreter_excluded_from_scalar_proxy=True)
    try:
        model=tuple(np.asarray(v,dtype=float) for v in model)
        if len(model)!=3:raise ValueError('Expected Wi,Wh,Wo')
        Wi,Wh,Wo=model;H=Wi.size
        add('model_validation',5*sum(v.size for v in model))
        if H<1 or Wi.shape!=(H,) or Wh.shape!=(H,H) or Wo.shape!=(10,H) or not all(np.isfinite(v).all() for v in model):
            raise ValueError('Expected finite T10 bias-free model')
        budget=float(budget)
        if not math.isfinite(budget) or budget<=0:raise ValueError('Finite positive budget required')
        counts=(max(8,int(.66*budget/8)),max(8,int(.28*budget/7.1)),max(11,int(.06*budget)))
        if max(counts)>65536:raise ValueError('Requested node batch exceeds65536')
        stage='setup';text=SOURCE.read_text()
        if hashlib.sha256(text.encode()).hexdigest()!=EXPECTED:raise RuntimeError('Shared tube source changed')
        add('cold_geometry_tables',500*(200001+20001))
        module=types.ModuleType('counted_uniform_tube_geometry');exec(compile(text,str(SOURCE),'exec'),module.__dict__)
        oldq=module._qmc
        def qmc(n,d,skip=0):
            if skip:raise ValueError('Frozen port uses no Sobol skip')
            add('sobol_nodes',100000*d+30*d*n);add('node_shift_remainder',4*d*n)
            kernels['sobol_initializations']=kernels.get('sobol_initializations',0)+1
            kernels['sobol_coordinates']=kernels.get('sobol_coordinates',0)+d*n
            nodes=oldq(n,d,skip);digest.update(np.asarray(nodes,dtype='<f8').tobytes());return nodes
        module._qmc=qmc;parts={};old_fpe=0.;stage='estimation'
        for kind,n,face in (('23',counts[0],module.faces23),('12',counts[1],module.faces12)):
            add('face_geometry',n*(1000+230*9+80));add('radius_direction',n*(300+20*10));add('aggregation',1000+10*n)
            kernels['face_quantile_coordinates']=kernels.get('face_quantile_coordinates',0)+n
            kernels['face_normal_transform_coordinates']=kernels.get('face_normal_transform_coordinates',0)+8*n
            fc=face(n,q=None)
            for key in ('x0','d','lo','hi','tpos','tneg'):face_digest.update(np.asarray(fc[key],dtype='<f8').tobytes())
            if legacy:
                mass,remainder,oldcost=module.F_faces(model,fc,400,radial=True)
                old_fpe+=float(oldcost.sum())
                records.append(dict(kind=kind,rows=n,legacy_remaining_mass=float(remainder.sum()),legacy_fpe=float(oldcost.sum())))
                if np.any(remainder>0):raise RuntimeError('Legacy unintegrated radial remainder; no completed estimate')
            else:
                rho=np.linalg.norm(fc['x0'],axis=1)
                if np.any(rho<=0) or not np.isfinite(rho).all():raise ArithmeticError('Invalid radial face norm')
                a=fc['d'].argmax(1);b=fc['d'].argmin(1)
                tau=np.maximum(fc['hi'],0)/(rho*math.sqrt(2));tex=fc['lo']/(rho*math.sqrt(2))
                try:
                    mass,_,ledger=walk_mass(model,fc['x0']/rho[:,None],a,b,tau,tex,top=kind=='12',details=True)
                except WalkFailure as exc:
                    failed_walk=exc.ledger;add('radial_native_walker',failed_walk['known_flops']);raise
                add('radial_native_walker',ledger['known_flops'])
                records.append(dict(kind=kind,rows=n,walker=ledger))
            parts['e'+kind]=float(fc['Z']*mass.mean())
        nb=counts[2];add('blind_transform_geometry',nb*(230*10+1000))
        add('blind_forward_and_ranking',nb*(10*(2*H*H+2*H)+10*(2*H-1)+100+9))
        add('aggregation',1000+10*nb)
        xb=module.blind_points(nb);face_digest.update(np.asarray(xb,dtype='<f8').tobytes())
        parts['blind']=float(module.P_BLIND*np.mean(module.forward_from(model,xb).argmax(1)!=np.argsort(xb,axis=1)[:,-2]))
        value=1-sum(parts.values())
        if not math.isfinite(value):raise ArithmeticError('Nonfinite estimate')
    except Exception as exc:
        ledger=snapshot(False);ledger.update(failed_walk=failed_walk,failure_kind='invalid_input' if stage=='validation' else 'numerical_or_implementation',
            cost_is_partial=stage!='validation',unresolved_failed_kernel_cost=failed_walk.get('unresolved_failed_kernel_cost',True) if failed_walk else stage!='validation')
        raise WalkFailure(str(exc),ledger) from exc
    ledger=snapshot(True);ledger.update(parts=parts,node_counts=counts,requested_budget=budget,
        fpe_unit=CORE.fpe_flops(H,10),actual_fpe=sum(costs.values())/CORE.fpe_flops(H,10),legacy_walk_fpe=old_fpe)
    return value,ledger['actual_fpe'],ledger

def est(model,budget,rng):
    value,cost,_=estimate(model,budget)
    return value,cost
