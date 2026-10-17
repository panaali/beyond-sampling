"""Deterministic T10 port with explicit cold scalar-operation PROXY.

Opaque Sobol/RNG/special-function charges are conventions, not proven bounds.
The public est always constructs a cold engine. No source is modified.
"""
import hashlib,math,time,types
from pathlib import Path
import numpy as np
from .native import walk_mass
from .adapter import CORE,WalkFailure

ROOT=Path('/workspace/clean-neurons/m16/approaches/sci_feynman')
HASHES={'core.py':'99ec9408e042f019a54dd34c399840ddeb3301c533a0bd11e5ec0a426e154037',
        'feyn.py':'ab568a48b90891302fa51e95c230a888d47f28ae221141730d9afac7e20b1309'}
TABLE_POINTS=400001
FULL_COST_CERTIFIED=False

class Engine:
    def __init__(self,legacy=False):
        start=time.process_time();self.legacy=legacy;self.used=False
        sources={name:(ROOT/name).read_text() for name in HASHES}
        for name,text in sources.items():
            if hashlib.sha256(text.encode()).hexdigest()!=HASHES[name]:
                raise RuntimeError('Shared source changed: '+name)
        self.core=types.ModuleType('feynman_port_geometry')
        exec(compile(sources['core.py'],str(ROOT/'core.py'),'exec'),self.core.__dict__)
        proxy=types.SimpleNamespace(**self.core.__dict__)
        for name in ('q23','q12','tn_above','tn_below'):
            original=getattr(self.core,name)
            def wrapped(u,*args,_name=name,_original=original):
                n=np.asarray(u).size
                if _name.startswith('q'):
                    self.add('quantile_interpolation',80*n);self.kernel('interp_queries',n)
                else:
                    self.add('truncated_normal',230*n);self.kernel('ndtri_arguments',n)
                    self.kernel('ndtr_arguments',np.asarray(args[0]).size)
                return _original(u,*args)
            setattr(proxy,name,wrapped)
        proxy.walk_mass=self.walk;proxy.forward=self.forward
        self.namespace={'C':proxy}
        source=sources['feyn.py']
        if source.count('import core as C')!=1:raise RuntimeError('Unexpected source import')
        exec(compile(source.replace('import core as C',''),str(ROOT/'feyn.py'),'exec'),self.namespace)
        self.original_points=self.namespace['unit_points'];self.namespace['unit_points']=self.points
        self.setup_cpu=time.process_time()-start

    def add(self,key,value):self.costs[key]=self.costs.get(key,0)+int(value)
    def kernel(self,key,value):self.kernels[key]=self.kernels.get(key,0)+int(value)

    def points(self,design,n,dim,key,shift_seed,ntop=2):
        if design!='sobol' or shift_seed is not None:raise ValueError('Only frozen deterministic Sobol supported')
        if n>65536:raise ValueError('Node batch exceeds65536')
        self.add('configuration_and_aggregation',1000+30*10*n)
        self.add('fixed_shift_rng_crc',20000+64*dim)
        self.kernel('rng_initializations',1);self.kernel('uniform_variates',dim)
        if n>1:
            grid=1<<(n-1).bit_length()
            self.add('sobol',100000*dim+30*dim*grid)
            self.kernel('sobol_initializations',1);self.kernel('sobol_generated_coordinates',dim*grid)
        else:self.add('constant_node_fill',dim)
        self.add('node_shift_remainder',4*dim*n)
        result=self.original_points(design,n,dim,key,shift_seed,ntop)
        self.nodes.update(key.encode());self.nodes.update(np.asarray(result,dtype='<f8').tobytes())
        self.node_rows+=n
        return result

    def walk(self,model,y,a,b,tau,tex,top=False,**kwargs):
        self.walk_calls+=1
        self.faces.update(np.asarray(y,dtype='<f8').tobytes())
        for array in (a,b,tau,tex):self.faces.update(np.asarray(array,dtype='<f8').tobytes())
        if self.legacy:
            result=self.core.walk_mass(model,y,a,b,tau,tex,top=top,**kwargs)
            self.legacy_walk_fpe+=float(result[1].sum());return result
        try:
            mass,cost,ledger=walk_mass(model,y,a,b,tau,tex,top=top,details=True,**kwargs)
        except WalkFailure as exc:
            self.add('native_walker',exc.ledger['known_flops'])
            self.failed_walk=exc.ledger
            raise
        self.add('native_walker',ledger['known_flops'])
        self.shared_calls+=ledger['shared_calls']
        self.interval_count+=ledger['native_interval_counters']['normal_intervals']
        self.midpoint_bound+=ledger['native_interval_counters']['midpoint_quadrature_bound']
        return mass,cost

    def forward(self,model,x):
        H=len(model[0]);T=10;n=len(x)
        self.add('blind_forward_and_ranking',n*(T*(2*H*H+2*H)+T*(2*H-1)+T*T+T-1))
        self.kernel('blind_inputs',n)
        return self.core.forward(model,x)

    def estimate(self,model,budget):
        start=time.process_time();cold=not self.used;self.used=True
        self.costs={};self.kernels={};self.nodes=hashlib.sha256();self.faces=hashlib.sha256()
        self.node_rows=self.walk_calls=self.shared_calls=self.interval_count=0
        self.midpoint_bound=self.legacy_walk_fpe=0.;self.failed_walk=None
        self.add('cold_geometry_table',500*TABLE_POINTS if cold else 0)
        if cold:
            self.kernel('geometry_table_points',TABLE_POINTS)
            self.kernel('table_pdf_arguments',2*TABLE_POINTS)
            self.kernel('table_cdf_arguments',2*TABLE_POINTS)
            self.kernel('table_sf_arguments',TABLE_POINTS)
        self.add('validation_and_final_aggregation',1000)
        try:
            model=tuple(np.asarray(v,dtype=float) for v in model)
            if len(model)!=3:raise ValueError('Expected Wi,Wh,Wo')
            Wi,Wh,Wo=model;H=Wi.size
            self.add('model_validation',5*sum(v.size for v in model))
            if Wi.shape!=(H,) or H<1 or Wh.shape!=(H,H) or Wo.shape!=(10,H) or not all(np.isfinite(v).all() for v in model):
                raise ValueError('Expected finite bias-free T10 model')
            budget=float(budget)
            if not math.isfinite(budget) or budget<=0:raise ValueError('Budget must be finite and positive')
            n23=max(1,int(budget*.70/(360*7.5)));n12=max(1,int(budget*.26/(45*6.7)));nb=max(1,int(budget*.04/11))
            if max(n23,n12,nb)>65536:raise ValueError('Requested node batch exceeds65536')
            value,old_cost,parts=self.namespace['estimate'](model,budget)
            if not math.isfinite(value):raise ArithmeticError('Nonfinite estimate')
        except Exception as exc:
            ledger=self.ledger(cold,start,False)
            ledger['failure_kind']='walker' if isinstance(exc,WalkFailure) else 'invalid_input' if isinstance(exc,(ValueError,TypeError)) else 'numerical_or_implementation'
            ledger['failed_walk']=self.failed_walk
            ledger['cost_is_partial']=self.failed_walk is not None and self.failed_walk.get('cost_is_partial',True)
            raise WalkFailure(str(exc),ledger) from exc
        ledger=self.ledger(cold,start,True)
        ledger.update(parts=parts,requested_budget=budget,legacy_reported_fpe=float(old_cost),
                      actual_fpe=ledger['known_flops']/CORE.fpe_flops(H,10),
                      fpe_unit=CORE.fpe_flops(H,10),legacy_walk_fpe=self.legacy_walk_fpe)
        return float(value),ledger['actual_fpe'],ledger

    def ledger(self,cold,start,complete):
        return dict(complete=complete,known_flops=sum(self.costs.values()),components=dict(self.costs),kernels=dict(self.kernels),
                    scalar_cost_is_proxy=True,portable_bound_certified=False,opaque_kernel_costs_certified=False,
                    legacy_walker_cost_missing_from_proxy=self.legacy,source_hashes=HASHES,cold=cold,
                    saved_warm_setup_flops=0 if cold else 500*TABLE_POINTS,
                    cpu_seconds=time.process_time()-start+(self.setup_cpu if cold else 0.),
                    setup_cpu_seconds=self.setup_cpu if cold else 0.,node_sha256=self.nodes.hexdigest(),face_sha256=self.faces.hexdigest(),
                    node_rows=self.node_rows,walk_calls=self.walk_calls,shared_walk_calls=self.shared_calls,
                    normal_intervals=self.interval_count,midpoint_quadrature_bound=self.midpoint_bound,
                    roundoff_certified=False,imports_and_interpreter_excluded_from_scalar_proxy=True)

def est(model,budget,rng):
    """ARC interface, fixed nodes independent of rng; cost includes fresh table setup."""
    value,cost,_=Engine().estimate(model,budget)
    return value,cost
