"""Stochastic boundary baseline with completed A/B lines and explicit cold costs."""
import hashlib,math,time,types
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,ndtri
from scipy.stats import norm
from .laplace_estimator import noimports
from .native import walk_mass
from .adapter import CORE,WalkFailure
ROOT=Path('/workspace/clean-neurons/m16/approaches/boundary')
PINS={'zones.py':'d55a86b24532558b7803e850062fb586281e2768f302f775145d0ed7c3023c01',
'zones2.py':'71c3d8398051deba76e9039974a3d7bb4caab263a72d44e7855251da62fbd905',
'direct.py':'2c0c2b456ea8819ae4bafded43566e90d31ba9d6c7e1b373014c76950f7431f1',
'estimator.py':'8f9ee8cb0001a433059da50061e94496abf86c15a101bd77aa62b8c7ce9cf31a'}
DEDUCTIVE=False

class Engine:
    def __init__(self,rng):
        self.start=time.process_time();self.cost={'validation_and_allocation':11000};self.counts={};self.records=[];self.failed_walk=None
        outer=self
        class CountedRNG:
            def random(self,*args,**kwargs):return self.draw('random',64,*args,**kwargs)
            def standard_normal(self,*args,**kwargs):return self.draw('standard_normal',100,*args,**kwargs)
            def integers(self,*args,**kwargs):return self.draw('integers',50,*args,**kwargs)
            def draw(self,method,charge,*args,**kwargs):
                result=getattr(rng,method)(*args,**kwargs);n=np.asarray(result).size
                outer.add('rng_'+method,charge*n);outer.counts[method]=outer.counts.get(method,0)+n;return result
        self.rng=CountedRNG()
        def counted_quad(fun,*args,**kwargs):
            self.add('normalizer_setup',1000)
            def counted(x):
                self.add('normalizer_integrands',1000);self.counts['quad_evaluations']=self.counts.get('quad_evaluations',0)+1
                return fun(x)
            return quad(counted,*args,**kwargs)
        common=dict(np=np,ndtr=ndtr,ndtri=ndtri,norm=norm,quad=counted_quad)
        texts={n:(ROOT/n).read_text() for n in PINS}
        if any(hashlib.sha256(texts[n].encode()).hexdigest()!=h for n,h in PINS.items()):raise RuntimeError('Boundary source changed')
        self.Z=types.SimpleNamespace();z=dict(common);exec(noimports(texts['zones.py']),z);self.Z.__dict__.update(z)
        self.add('cold_inverse_tables',500*200001)
        d=dict(common);exec(noimports(texts['direct.py']),d);self.D=types.SimpleNamespace(**d)
        z2=dict(common,Z=self.Z,D=self.D);exec(noimports(texts['zones2.py']),z2);self.Z2=types.SimpleNamespace(**z2)
        self.entry=dict(common,Z=self.Z,Z2=self.Z2,D=self.D);exec(noimports(texts['estimator.py']),self.entry);self.entry['_run']=self.run
    def add(self,k,v):self.cost[k]=self.cost.get(k,0)+int(v)
    def snapshot(self,complete):return dict(complete=complete,known_flops=sum(self.cost.values()),components=self.cost.copy(),counts=self.counts.copy(),records=self.records,
        source_hashes=PINS,cpu_seconds=time.process_time()-self.start,scalar_cost_is_proxy=True,opaque_kernel_costs_certified=False,
        portable_bound_certified=False,roundoff_certified=False,deductive=False,completed_line_variant=True,
        original_stochastic_remainder_replaced=True,adaptive_pilot_reuse_unbiasedness_not_asserted=True,cold=True)
    def lines(self,model,G):
        y=G['y'];n=len(y);a=G['d'].argmax(1);b=G['d'].argmin(1)
        top=bool(np.all(G['tp']==b))
        if not(top or np.all(G['tp']==a)):raise ValueError('Mixed target convention')
        tau=np.maximum(np.minimum(G['tau'],G['mhi']),0.)
        try:mass,_,ledger=walk_mass(model,y,a,b,tau,G['mlo'],top=top,details=True)
        except WalkFailure as exc:self.failed_walk=exc.ledger;self.add('native_lines',exc.ledger['known_flops']);raise
        self.add('native_lines',ledger['known_flops']);self.counts['surface_model_lines']=self.counts.get('surface_model_lines',0)+n
        return G['w']*mass,ledger
    def zone_draw(self,n,cap=1000000):
        output=[];got=0;generated=0
        while got<n:
            count=min(65536,max(4*n,1000))
            if generated+count>cap:raise RuntimeError('Explicit zone candidate cap exhausted')
            x=self.rng.standard_normal((count,10));self.add('zone_classification',500*count)
            z=self.Z.zone_of(x,self.entry['GA'],self.entry['GB']);accepted=x[z==3]
            output.append(accepted);got+=len(accepted);generated+=count
            self.counts['rejection_candidates']=self.counts.get('rejection_candidates',0)+count
        self.counts['discarded_or_overshoot_candidates']=self.counts.get('discarded_or_overshoot_candidates',0)+generated-n
        return np.concatenate(output)[:n],generated
    def run(self,model,kind,n,rng):
        if n<=0:return np.zeros(0),0.
        if n>65536:raise ValueError('Requested batch exceeds65536')
        before=sum(self.cost.values());H=len(model[0]);unit=CORE.fpe_flops(H,10);self.add('term_aggregation',1000+10*n)
        if kind in ('A','B'):
            self.add('surface_geometry',n*(1000+230*9+80))
            G=self.Z2.geom_A(model,n,self.rng,self.entry['GA']) if kind=='A' else self.Z2.geom_B(model,n,self.rng,self.entry['GA'],self.entry['GB'])
            values,ledger=self.lines(model,G);record=dict(kind=kind,rows=n,line_ledger=ledger)
        else:
            if kind=='blind':
                self.add('blind_transform_geometry',1000*n);x=self.D.blind(n,self.rng);generated=n
            else:x,generated=self.zone_draw(n)
            self.add('point_forward_and_ranking',n*(10*(2*H*H+2*H)+10*(2*H-1)+100+9))
            values=self.Z.point_wrong(model,x).astype(float);self.counts['direct_model_inputs']=self.counts.get('direct_model_inputs',0)+n
            record=dict(kind=kind,rows=n,generated=generated)
        charge=sum(self.cost.values())-before;record['flops']=charge;self.records.append(record);return values,charge/unit
    def estimate(self,model,budget):
        stage='validation'
        try:
            model=tuple(np.asarray(w,dtype=float) for w in model)
            if len(model)!=3:raise ValueError('Expected three matrices')
            Wi,Wh,Wo=model;H=Wi.size;self.add('model_validation',5*sum(w.size for w in model))
            if H<1 or Wi.shape!=(H,) or Wh.shape!=(H,H) or Wo.shape!=(10,H) or not all(np.isfinite(w).all() for w in model):raise ValueError('Finite T10 model required')
            budget=float(budget)
            if not math.isfinite(budget) or budget<=0:raise ValueError('Finite positive budget required')
            stage='estimation';value,_,terms,n,costs=self.entry['est_boundary'](model,budget,self.rng,details=True)
            if not math.isfinite(value):raise ArithmeticError('Nonfinite estimate')
        except Exception as exc:
            result=self.snapshot(False);result.update(failure_kind='invalid_input' if stage=='validation' else 'numerical_or_implementation',failed_walk=self.failed_walk,
                cost_is_partial=stage!='validation',unresolved_failed_kernel_cost=self.failed_walk.get('unresolved_failed_kernel_cost',True) if self.failed_walk else stage!='validation')
            raise WalkFailure(str(exc),result) from exc
        result=self.snapshot(True);result.update(terms=terms,term_rows=n,term_fpe=costs,requested_budget=budget,
            fpe_unit=CORE.fpe_flops(H,10),actual_fpe=sum(self.cost.values())/CORE.fpe_flops(H,10))
        return float(value),result['actual_fpe'],result

def est(model,budget,rng):
    value,cost,_=Engine(rng).estimate(model,budget);return value,cost
