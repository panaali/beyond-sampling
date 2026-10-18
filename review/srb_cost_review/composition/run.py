"""Deterministic API controls, not samples from the Gaussian target law."""
from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,json,math,sys,time,resource,warnings,importlib.util
import numpy as np
P=Path(__file__).resolve().parent
FILENAME='<frozen-srb-composition>'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def array(a):
    a=np.asarray(a);d=dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
    if a.size<=10000:d['values']=[float(v).hex() if a.dtype.kind=='f' else int(v) for v in a.ravel()]
    return d
class Stream:
    def __init__(self,prototypes,delayed):self.prototypes=prototypes;self.delayed=delayed;self.requests=[]
    def standard_normal(self,shape):
        assert shape==(50000,9)
        k=len(self.requests);assert k<(2 if self.delayed else 1),'unexpected extra batch'
        cycle=6 if self.delayed and k==0 else 7
        x=self.prototypes[np.arange(50000)%cycle].copy()
        self.requests.append(dict(shape=list(shape),cycle=cycle,sha256=hashlib.sha256(x.tobytes()).hexdigest(),candidate_count=50000))
        return x

def extraction(proto,originalQ):
    src=Path(proto['cv_source']).read_text();tree=ast.parse(src)
    names={'pstratum','pc','draw_all','make'};constants={'T','EDGES','NS','_pc','COST_DRAW'}
    body=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets))]
    assert {n.name for n in body if isinstance(n,ast.FunctionDef)}==names
    assert {t.id for n in body if isinstance(n,ast.Assign) for t in n.targets}==constants
    env=dict(np=np,Q=originalQ,COST_Q=1.6,PRE=proto['precompute'])
    mod=ast.Module(body=body,type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,FILENAME,'exec'),env)
    assert env['T']==10 and env['NS']==7 and env['COST_DRAW']==.1 and env['_pc'] is None
    return env,ast.unparse(mod)

def charging(proto):
    root=ast.parse(Path(proto['harness_source']).read_text());fn=next(n for n in root.body if isinstance(n,ast.FunctionDef) and n.name=='_charged')
    overrides=next(n.value for n in root.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OVERRIDES' for t in n.targets))
    assert isinstance(overrides,ast.Dict)
    value=next(v for k,v in zip(overrides.keys,overrides.values) if isinstance(k,ast.Constant) and k.value=='control_variate')
    env={};entry=eval(compile(ast.Expression(value),'<frozen-srb-override>','eval'),env)
    env['OVERRIDES']={'control_variate':entry};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen-charge>','exec'),env)
    return env['_charged'],entry,ast.unparse(fn),ast.unparse(value)

def observe(frame,event,arg,records):
    if frame.f_code.co_filename!=FILENAME:return None
    frame.f_trace_lines=False;frame.f_trace_opcodes=False
    if event=='return':
        name=frame.f_code.co_name;loc=frame.f_locals
        if arg is None:
            records.append(dict(function=name,exceptional_or_none_return=True));return None
        if name=='pstratum':records.append(dict(function=name,input=array(loc['x']),output=array(arg),strata_counts=np.bincount(arg,minlength=7).tolist()))
        elif name=='draw_all':records.append(dict(function=name,nh=array(loc['nh']),got=array(loc['got']),drawn=loc['drawn'],returned=[array(x) for x in arg[0]],returned_drawn=arg[1]))
        elif name=='pc':records.append(dict(function=name,keys=sorted(arg),P=array(arg['P'])))
        elif name=='est':records.append(dict(function=name,c_hex=float(loc['c']).hex(),n=loc['n'],nh=array(loc['nh']),x=array(loc['x']),h=array(loc['h']),w=array(loc['w']),q=array(loc['q']),estimate_hex=float(arg[0]).hex(),cost=arg[1]))
    return lambda f,e,a:observe(f,e,a,records)

def main():
    start=time.process_time();out=P/'results.json';progress=P/'progress.jsonl';trees=P/'trees.json'
    assert not any(p.exists() for p in (out,progress,trees))
    result=dict(complete=False,rows=[],new_random_Gaussian_trained_MC_calls=0)
    def log(x):
        with progress.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
    def save():result.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);write(out,result)
    log(dict(stage='started'))
    try:
        proto=json.loads((P/'protocol.json').read_text());gate=json.loads((P/'gate.json').read_text());assert gate['complete'] and gate['protocol_sha256']==sha(P/'protocol.json')
        for p,h in {**proto['pins'],**gate['pins']}.items():assert sha(p)==h,p
        result.update(protocol_sha256=sha(P/'protocol.json'),gate_sha256=sha(P/'gate.json'),scope=proto['scope'],runtime=dict(python=sys.version,numpy=np.__version__))
        spec=importlib.util.spec_from_file_location('prior_q_loader',proto['q_loader']);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
        qenv,qtree=loader.load(Path(proto['q_source']).read_text());Q=qenv['Q']
        charged,override,charge_tree,override_tree=charging(proto)
        fixtures=json.loads(Path(proto['q_fixtures']).read_text());toy=next(r for r in fixtures if r['name']=='zero_recurrent');model=tuple(np.array(toy[k],dtype=np.float64) for k in ('Wi','Wh','Wo'))
        alloc=np.array(json.loads(Path(proto['allocation']).read_text())['plain'],dtype=np.float64)
        setup=json.loads((P/'fixtures.json').read_text());prototypes=np.array(setup['prototypes'],dtype=np.float64);assert prototypes.shape==(7,9)
        env,ctree=extraction(proto,Q);write(trees,dict(Q=qtree,composition=ctree,charged=charge_tree,override=override_tree))
        # One explicit seven-row API control, separate from eight estimator calls.
        prototype_labels=env['pstratum'](prototypes);result['prototype_labels']=prototype_labels.tolist();save();assert np.array_equal(prototype_labels,np.arange(7))
        for case in setup['cases']:
            if time.process_time()-start>16:raise RuntimeError('16CPU soft cap; no automatic retry')
            row=dict(name=case['name'],budget=case['budget'],delayed=case['delayed'],arms=[]);result['rows'].append(row);log(dict(stage='case_started',name=case['name']));save()
            for mode in ('original','observed'):
                log(dict(stage='arm_started',name=case['name'],mode=mode));env,tree=extraction(proto,Q);assert tree==ctree
                est=env['make'](strat=True,cv=None,alloc=alloc);rng=Stream(prototypes,case['delayed']);records=[]
                oldtrace=sys.gettrace();assert oldtrace is None
                arm=dict(mode=mode);row['arms'].append(arm);save()
                score=cost=None;ws=[]
                try:
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always')
                        if mode=='observed':sys.settrace(lambda f,e,a:observe(f,e,a,records))
                        score,cost=est(model,case['budget'],rng)
                finally:
                    sys.settrace(oldtrace)
                    arm.update(requests=rng.requests,records=records,warnings=sorted((w.category.__name__,str(w.message)) for w in ws))
                    if score is not None and cost is not None:arm.update(estimate_hex=float(score).hex(),cost=cost)
                    if env.get('_pc') is not None:arm['cacheP']=array(env['_pc']['P'])
                    save()
                arm.update(estimate_hex=score.hex(),cost=cost,warnings=sorted((w.category.__name__,str(w.message)) for w in ws),requests=rng.requests,records=records,cacheP=array(env['_pc']['P']));save()
                assert sys.gettrace() is oldtrace
                assert len(rng.requests)==(2 if case['delayed'] else 1)
                log(dict(stage='arm_completed',name=case['name'],mode=mode))
            a,b=row['arms'];row['parity']=all(a[k]==b[k] for k in ('estimate_hex','cost','warnings','requests','cacheP'));save();assert row['parity']
            es=[r for r in b['records'] if r['function']=='est'];ds=[r for r in b['records'] if r['function']=='draw_all'];ps=[r for r in b['records'] if r['function']=='pc'];hs=[r for r in b['records'] if r['function']=='pstratum']
            assert len(es)==len(ds)==len(ps)==1 and len(hs)==len(b['requests'])
            e,d=es[0],ds[0];c=1.6+.1;n=int(case['budget']//c);nh=np.maximum(4,np.floor(alloc*n).astype(int));M=int(nh.sum());D=50000*len(b['requests'])
            assert e['c_hex']==c.hex() and e['n']==n and e['nh']==array(nh) and e['x']['shape']==[M,9]
            assert d['nh']==d['got']==array(nh) and d['drawn']==d['returned_drawn']==D
            assert a['cost']==math.ceil(M*c) and e['cost']==a['cost']
            old=charged('control_variate',a['cost'],case['budget']);hypothetical=math.ceil(M*(3.3+.1));extra=F(D-M,25)
            row.update(n=n,nh=nh.tolist(),kept=M,drawn=D,raw_cost=a['cost'],harness_charged_hex=float(old).hex(),override_factor_hex=float(override['cost_factor']).hex(),hypothetical_Q_only_charge=hypothetical,literal_extra_candidate_rule=str(extra),reported_draw_allowance=str(F(M,10)),policy_comparison_scope='Nominal rule5 treats each kept prefix as one free input evaluation; mock stream has no Gaussian-law or RNG-generation cost inference. Q-only3.3 replacement is hypothetical, not a validated new count.')
            save();log(dict(stage='case_completed',name=case['name'],M=M,D=D))
        assert len(result['rows'])==4 and sum(len(r['arms']) for r in result['rows'])==8
        result.update(complete=True,estimator_calls=8,prototype_stratum_control_calls=1,full_cost=False,probabilistic_sampling_validation=False)
        save();log(dict(stage='completed',sha256=sha(out)));print(json.dumps(dict(complete=True,cpu=result['cpu_seconds'],rows=[dict(name=r['name'],M=r['kept'],D=r['drawn']) for r in result['rows']])))
    except BaseException as e:result['failure']=repr(e);save();log(dict(stage='failure',error=repr(e)));raise
if __name__=='__main__':main()
