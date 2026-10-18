"""Source-preserving prefix stop, explicit lookahead and resource observer."""
from pathlib import Path
import importlib.util,json,hashlib,time,sys
import numpy as np
H=Path(__file__).resolve().parent

class Budget(Exception):pass

def load_engine(protocol):
    for p,h in protocol['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    source=Path(protocol['m43']);sys.path.insert(0,str(source.parent))
    spec=importlib.util.spec_from_file_location('t188_frozen_engine',H/'estT_snapshot.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
    assert Path(sys.modules['m43'].__file__).resolve()==source
    return E

def hexarray(a):return [[float(x).hex() for x in row] for row in np.asarray(a)]
def cone_pack(c):return dict(R=hexarray(c.R),N=hexarray(c.N),T=c.T.astype(int).tolist())
def cone_load(E,c):return E.Cone(np.array([[float.fromhex(x) for x in r] for r in c['R']]),np.array([[float.fromhex(x) for x in r] for r in c['N']]),np.array(c['T'],dtype=bool))

class Observer:
    def __init__(self,E,record):
        self.E=E;self.record=record;self.start=time.process_time();self.proxy=0;self.finish_calls=0;self.lookahead_calls=0;self.lookahead_points=0;self.states=[];self.stack=[];self.parent=None;self.mass_calls=0
        owner=self
        class Ops:
            def __init__(self):self.n=0
            def add(self,n):
                self.n+=n;owner.proxy+=n;owner.check()
        self.Ops=Ops;E.Ops=Ops
        original=E._finish;original_lookahead=E._frozen_margin
        def forbidden(*args,**kwargs):owner.mass_calls+=1;raise AssertionError('unexpected final-cell or Gaussian mass call')
        E.mass=forbidden;E.orthant=forbidden;E._orth5=forbidden;E._cell_error=forbidden
        def lookahead(cmodel,t,A,g,truth,P,ops):
            owner.lookahead_calls+=1;owner.lookahead_points+=len(P);owner.check()
            return original_lookahead(cmodel,t,A,g,truth,P,ops)
        E._frozen_margin=lookahead
        def stop(c,g,A,t,model,truth,n_true,eps,ops,exact):
            owner.check();owner.finish_calls+=1;T=model[2].shape[0]
            assert np.all((g==0)|(g==1)) and np.all(np.isfinite(A))
            owner.stack.append(tuple(int(x) for x in g))
            try:
                assert len(owner.stack)==t and 1<=t<=T-1
                if t==T-1:
                    assert np.all(A[:,t:]==0)
                    row=dict(parent=owner.parent,truth=int(truth),depth=t,history=[list(x) for x in owner.stack],g=[int(x) for x in g],A=hexarray(A),state=hexarray(g[:,None]*A),cone=cone_pack(c),n_true=[float(x).hex() for x in n_true])
                    owner.states.append(row);owner.record(dict(stage='prefix_state',row=row));return 0.0
                return original(c,g,A,t,model,truth,n_true,eps,ops,exact)
            finally:owner.stack.pop()
        E._finish=stop;E.MODE[0]='recipe108';E.EXACT_STEPS[0]=set();E.DIAG[0]=None
    def check(self):
        if self.proxy>5_000_000 or len(self.states)>10_000 or time.process_time()-self.start>12:raise Budget('12CPU/5M inheritedproxy/10000states cap')
    def run_parent(self,parent,h,model):
        self.parent=parent;self.check();assert not self.stack
        Wi,Wh,Wo=model;T=Wo.shape[0];g=(h['sigma']*Wi>0).astype(float);A=np.outer(Wi,np.eye(T)[0]);ops=self.Ops();before=len(self.states)
        self.record(dict(stage='parent_attempt',parent=parent,T=T));returned=self.E._finish(h['cone'],g,A,1,model,h['truth'],h['n_true'],.1,ops,False)
        assert returned==0.0 and not self.stack
        row=dict(parent=parent,T=T,states=len(self.states)-before,source_proxy=ops.n);self.record(dict(stage='parent_result',row=row));return row
    def summary(self):return dict(proxy=self.proxy,finish_calls=self.finish_calls,lookahead_calls=self.lookahead_calls,lookahead_points=self.lookahead_points,states=len(self.states),mass_or_final_calls=self.mass_calls,cpu_seconds=time.process_time()-self.start,stats=dict(self.E.STATS))
