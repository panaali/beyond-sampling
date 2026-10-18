"""Once-only exact finite-law controls. No model, distribution sampler or RNG."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,resource,sys,time,traceback

HERE=Path(__file__).resolve().parent
START=time.process_time()
OUT=HERE/'output'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify(pins):
    assert pins
    for p,h in pins.items():assert sha(p)==h,p
def atomic(p,x):
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)

def invalid(base,name,cases):
    x=copy.deepcopy(base)
    if name=='duplicate_state':x['rows'][1]['x']=x['rows'][0]['x'][:]
    elif name=='unnormalized_weights':x['rows'][0]['weight']=['1','3']
    elif name=='zero_weight':x['rows'][0]['weight']=['0','1']
    elif name=='negative_weight':x['rows'][0]['weight']=['-1','2']
    elif name=='not_reversal_closed':x['rows'][1]['x']=[2,-1]
    elif name=='surrogate_above_one':x['rows'][0]['surrogate']=['3','2']
    elif name=='nonbinary_correct':x['rows'][0]['correct']=2
    elif name=='boolean_correct':x['rows'][0]['correct']=True
    elif name=='zero_denominator':x['rows'][0]['weight']=['1','0']
    elif name=='zero_cost':x['paired_cost']=['0','1']
    elif name=='insufficient_budget':x['budget']=['1','1']
    elif name=='strict_noninvariant':
        x=copy.deepcopy(cases[7]);x['require_invariant']=True
    else:raise AssertionError('unfrozen guard')
    return x

def main():
    OUT.mkdir(exist_ok=False)
    r=dict(complete=False,pass_all=False,cases=[],labels=[],guards=[],analyze_attempts=0,analyze_returns=0,label_attempts=0,label_returns=0)
    log=(OUT/'progress.jsonl').open('x',buffering=1)
    def save():
        r.update(cpu_seconds=time.process_time()-START,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic(OUT/'results.json',r)
    def event(kind,data):
        log.write(json.dumps(dict(kind=kind,data=data,cpu=time.process_time()-START),separators=(',',':'))+'\n')
        if time.process_time()-START>17:raise TimeoutError('17 CPU internal cap')
    try:
        p=read(HERE/'protocol.json');g=read(HERE/'gate.json')
        r['runtime']=dict(invoked=sys.executable,resolved=str(Path(sys.executable).resolve()),cpu_limits=resource.getrlimit(resource.RLIMIT_CPU),as_limits=resource.getrlimit(resource.RLIMIT_AS));save()
        verify(p['pins']);verify(g['pins'])
        assert r['runtime']['invoked']==p['python'] and r['runtime']['resolved']==p['resolved_python']
        assert resource.getrlimit(resource.RLIMIT_CPU)==(20,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert all(os.environ.get(k)=='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'])
        for key in ['review_commit','publication']:
            assert isinstance(g[key],str) and 8<=len(g[key])<=40 and all(c in '0123456789abcdef' for c in g[key])
        review=read(g['review_path']);verify(review['pins']);assert all(review[k] is True for k in ['complete','pass_all','execution_allowed'])
        for name in ['primitive.py','controls.py','fixtures.json','protocol.json','PROTOCOL.md','launch.sh']:
            path=str(HERE/name);assert review['pins'][path]==sha(path)==g['pins'][path]
        assert g['pins'][g['review_path']]==sha(g['review_path'])
        prior=read(p['proof_review']);assert prior['complete'] is True and prior['pass_all'] is True;verify(prior['pins'])
        f=read(HERE/'fixtures.json');assert len(f['cases'])==13 and len(f['labels'])==6 and len(f['guards'])==12
        spec=importlib.util.spec_from_file_location('t266_finite_primitive',HERE/'primitive.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        event('prerequisites_passed',dict(protocol_sha=sha(HERE/'protocol.json'),gate_sha=sha(HERE/'gate.json')))
        for i,x in enumerate(f['cases']):
            r['analyze_attempts']+=1;save();event('case_attempt',dict(index=i,name=x['name']))
            value=m.analyze(x);r['analyze_returns']+=1
            row=dict(index=i,name=x['name'],input=x,returned=value,passed=False);r['cases'].append(row);save()
            for k,want in x['expected'].items():
                got=value['scalars'][k] if k in value['scalars'] else value[k]
                assert got==want,(x['name'],k,got,want)
            row['passed']=True;save();event('case_result',dict(index=i,name=x['name'],passed=True))
        for i,x in enumerate(f['labels']):
            r['label_attempts']+=1;save();event('label_attempt',dict(index=i,name=x['name']))
            value=m.labels(x);r['label_returns']+=1
            row=dict(index=i,name=x['name'],input=x,returned=value,passed=False);r['labels'].append(row);save()
            assert value==x['expected'];row['passed']=True;save();event('label_result',dict(index=i,name=x['name'],passed=True))
        for i,name in enumerate(f['guards']):
            x=invalid(f['cases'][0],name,f['cases']);r['analyze_attempts']+=1;save();event('guard_attempt',dict(index=i,name=name))
            try:value=m.analyze(x)
            except ValueError as e:
                row=dict(index=i,name=name,input=x,rejected=True,error=str(e));r['guards'].append(row);save();event('guard_result',dict(index=i,name=name,rejected=True))
            else:
                r['analyze_returns']+=1;r['guards'].append(dict(index=i,name=name,input=x,rejected=False,returned=value));save();raise AssertionError('guard returned')
        assert len(r['cases'])==13 and len(r['labels'])==6 and len(r['guards'])==12
        assert (r['analyze_attempts'],r['analyze_returns'],r['label_attempts'],r['label_returns'])==(25,13,6,6)
        r['work']=dict(table_rows=sum(x['returned']['table_rows'] for x in r['cases']),paired_residual_lookups=sum(x['returned']['paired_residual_lookups'] for x in r['cases']),exact_table_calls=13,label_calls=6,rejected_table_calls=12,model_calls=0,gaussian_calls=0,rng_calls=0)
        save();event('complete',r['work']);r['complete']=r['pass_all']=True;save()
    except BaseException:
        r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();log.write(json.dumps(dict(kind='fatal',data=r['failure']))+'\n');raise
    finally:log.close()

if __name__=='__main__':main()
