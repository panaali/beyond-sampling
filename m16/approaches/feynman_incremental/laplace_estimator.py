"""Pinned Laplace nodes, completed corrected near mass, explicit unresolved tails."""
import ast,hashlib,math,time,types
from pathlib import Path
import numpy as np
from .radial import walk_mass,radial_interval
from .adapter import CORE,WalkFailure
from .tube_estimator import SOURCE as TUBE,EXPECTED as TUBE_HASH
ROOT=TUBE.parent.parent/'sci_laplace'
PINS={'lp.py':'7372b3a980ee389db856a153f8210acc02b2032e2fd759e7392a61bb46d21b35',
'lqt.py':'55985f1adf35afcd44dcf8267be8cc921d33754a211030b6ac6f9a0aa0d150bc',
'estimator.py':'d5c7680850b1d0b04321226bf4523860cb6e4371cc10132e9537fa2f5eb5ee25'}

def noimports(text):
    tree=ast.parse(text);tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom)) and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func).startswith('sys.path.'))]
    return compile(ast.fix_missing_locations(tree),'<pinned-local-namespace>','exec')

def estimate(model,budget,legacy=False):
    start=time.process_time();cost={'validation':1000};kernels={};records=[];stage='validation';failure=None
    nodes=hashlib.sha256();faces=hashlib.sha256();tails=hashlib.sha256()
    def add(k,v):cost[k]=cost.get(k,0)+int(v)
    def forward(x,kind,sort=False):
        n=len(x);add(kind,n*(10*(2*H*H+2*H)+10*(2*H-1)+9+(100 if sort else 0)))
        return C.forward_from(model,x)
    def ledger(done):return dict(complete=done,known_flops=sum(cost.values()),components=dict(cost),kernels=kernels.copy(),records=records,
        node_sha256=nodes.hexdigest(),face_sha256=faces.hexdigest(),tail_sha256=tails.hexdigest(),cpu_seconds=time.process_time()-start,
        source_hashes=PINS|{'tube.py':TUBE_HASH},scalar_cost_is_proxy=True,opaque_kernel_costs_certified=False,
        portable_bound_certified=False,roundoff_certified=False,legacy_walker_cost_missing_from_proxy=legacy,
        fixed_design_seed=0,cold=True,imports_and_interpreter_excluded_from_scalar_proxy=True)
    try:
        model=tuple(np.asarray(w,dtype=float) for w in model)
        if len(model)!=3:raise ValueError('Expected three matrices')
        Wi,Wh,Wo=model;H=Wi.size
        add('model_validation',5*sum(w.size for w in model))
        if H<1 or Wi.shape!=(H,) or Wh.shape!=(H,H) or Wo.shape!=(10,H) or not all(np.isfinite(w).all() for w in model):raise ValueError('Expected finite T10 model')
        budget=float(budget)
        if not math.isfinite(budget) or budget<=0:raise ValueError('Finite positive budget required')
        m23=max(1,round(.66*budget/(7.5*360)));m12=max(1,round(.27*budget/(8*45)));nb=max(11,int(.05*budget))
        if max(360*m23,45*m12,nb)>65536:raise ValueError('Batch limit65536 exceeded')
        stage='setup';source=TUBE.read_text()
        if hashlib.sha256(source.encode()).hexdigest()!=TUBE_HASH:raise RuntimeError('Tube source changed')
        texts={n:(ROOT/n).read_text() for n in PINS}
        if any(hashlib.sha256(texts[n].encode()).hexdigest()!=h for n,h in PINS.items()):raise RuntimeError('Laplace source changed')
        add('cold_geometry_tables',500*(200001+20001));C=types.ModuleType('laplace_geometry');exec(compile(source,str(TUBE),'exec'),C.__dict__)
        ns=dict(C.__dict__);ns['COMBO']=np.array(C.COMBO23)
        tree=ast.parse(texts['lp.py']);definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build23')
        exec(compile(ast.Module(body=[definition],type_ignores=[]),'<pinned-build23>','exec'),ns)
        exec(noimports(texts['lqt.py']),ns)
        original_sets=ns['_sets']
        def sets(nstrata,m,d,seed):
            add('sobol_rng_setup',100000*d+20000+64*d);add('sobol_coordinates',30*d*m)
            add('stratum_shifts',64*nstrata*d+4*nstrata*m*d)
            kernels['sobol_initializations']=kernels.get('sobol_initializations',0)+1
            kernels['uniform_shift_coordinates']=kernels.get('uniform_shift_coordinates',0)+nstrata*d
            value=original_sets(nstrata,m,d,seed);nodes.update(np.asarray(value,dtype='<f8').tobytes());return value
        ns['_sets']=sets
        original_qmc=C._qmc
        def blindqmc(n,d,skip=0):
            add('blind_node_generation',100000*d+20000+64*d+30*d*n+4*d*n)
            kernels['sobol_initializations']=kernels.get('sobol_initializations',0)+1
            value=original_qmc(n,d,skip);nodes.update(np.asarray(value,dtype='<f8').tobytes());return value
        C._qmc=blindqmc
        def blind_forward(m,x):
            faces.update(np.asarray(x,dtype='<f8').tobytes());add('blind_transform_geometry',len(x)*(230*10+1000));add('aggregation',1000+10*len(x))
            return forward(x,'blind_forward_and_ranking',True)
        ns['forward_from']=blind_forward
        def hybrid(m,fc,k0,v_tail):
            nonlocal failure
            n=len(fc['x0']);add('face_geometry',n*(1000+230*9+80));add('radius_direction',n*500);add('aggregation',1000+10*n)
            for key in ('x0','d','lo','hi','tpos','tneg'):faces.update(np.asarray(fc[key],dtype='<f8').tobytes())
            rho=np.linalg.norm(fc['x0'],axis=1)
            if np.any(rho<=0) or not np.isfinite(rho).all():raise ArithmeticError('Invalid face radius')
            x=fc['x0']/rho[:,None];lo=fc['lo']/rho;hi=fc['hi']/rho
            a=fc['d'].argmax(1);b=fc['d'].argmin(1);top=bool(np.all(fc['tpos']==b))
            if not(top or np.all(fc['tpos']==a)):raise ValueError('Mixed target convention')
            if legacy:
                value,old=C.F_hybrid(m,fc,k0,v_tail)
                # Record the actual fixed tail nodes independently of old result bookkeeping.
                for side,sg in ((0,1.),(1,-1.)):
                    left=np.maximum(lo,k0);active=hi>left
                    ga,gb=C.Grad(left[active]),C.Grad(hi[active]);k=C.Ginv(ga+v_tail[active,side]*(gb-ga))
                    tails.update(np.asarray(x[active]+sg*k[:,None]*fc['d'][active],dtype='<f8').tobytes())
                records.append(dict(kind='12' if top else '23',rows=n,legacy_fpe=float(old.sum()),legacy_near_cap_uncertainty_unavailable=True))
                return value,old
            try:
                near,_,nl=walk_mass(m,x,a,b,np.maximum(np.minimum(hi,k0),0)/math.sqrt(2),lo/math.sqrt(2),top=top,details=True)
            except WalkFailure as exc:failure=exc.ledger;add('near_native_walker',failure['known_flops']);raise
            add('near_native_walker',nl['known_flops']);value=near.copy();tail_total=np.zeros(n);tail_selected=np.zeros(n)
            for side,sg,target in ((0,1.,fc['tpos']),(1,-1.,fc['tneg'])):
                left=np.maximum(lo,k0);active=hi>left;count=int(active.sum())
                if not count:continue
                add('tail_mass_and_nodes',count*(350+500+100+100))
                mass,_,_=radial_interval(left[active]/math.sqrt(2),hi[active]/math.sqrt(2))
                ga,gb=C.Grad(left[active]),C.Grad(hi[active]);k=C.Ginv(ga+v_tail[active,side]*(gb-ga))
                # Inverse-table interpolation has finite precision; do not clip an out-of-range node.
                if np.any(k<left[active]-1e-10) or np.any(k>hi[active]+1e-10):raise ArithmeticError('Tail node outside interval')
                inputs=x[active]+sg*k[:,None]*fc['d'][active];tails.update(np.asarray(inputs,dtype='<f8').tobytes())
                wrong=forward(inputs,'tail_forward_and_argmax').argmax(1)!=target[active]
                tail_total[active]+=mass;tail_selected[active]+=mass*wrong;value[active]+=mass*wrong
                kernels['tail_forward_inputs']=kernels.get('tail_forward_inputs',0)+count
            records.append(dict(kind='12' if top else '23',rows=n,near_mean=float(near.mean()),tail_mass_mean=float(tail_total.mean()),
                tail_selected_mean=float(tail_selected.mean()),near_walker=nl))
            return value,np.zeros(n)
        ns['F_hybrid']=hybrid;entry=dict(ns);exec(noimports(texts['estimator.py']),entry);stage='estimation'
        value,oldcost,parts=entry['est_seeded'](model,budget,seed=0,parts=True)
        if not math.isfinite(value):raise ArithmeticError('Nonfinite estimate')
    except Exception as exc:
        result=ledger(False);result.update(failure_kind='invalid_input' if stage=='validation' else 'numerical_or_implementation',failed_walk=failure,
            cost_is_partial=stage!='validation',unresolved_failed_kernel_cost=failure.get('unresolved_failed_kernel_cost',True) if failure else stage!='validation')
        raise WalkFailure(str(exc),result) from exc
    result=ledger(True);result.update(parts=parts,node_counts=(360*m23,45*m12,nb),requested_budget=budget,legacy_reported_fpe=float(oldcost),
        fpe_unit=CORE.fpe_flops(H,10),actual_fpe=sum(cost.values())/CORE.fpe_flops(H,10))
    if not legacy:
        z={'23':C.Z23,'12':C.Z12};near_error=sum(z[r['kind']]*r['near_mean'] for r in records)+parts['eb']
        tail_error=sum(z[r['kind']]*r['tail_mass_mean'] for r in records)
        result.update(conditional_tail_mass_bound=tail_error,conditional_accuracy_interval=[1-near_error-tail_error,1-near_error],
                      conditional_interval_excludes_outer_error=True)
    return float(value),result['actual_fpe'],result

def est(model,budget,rng):
    value,cost,_=estimate(model,budget)
    return value,cost
