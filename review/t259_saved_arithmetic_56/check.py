"""Fixed independent T259 saved arithmetic audit. No producer imports."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,os,re,resource,sys,time,traceback
from fixtures import inventory
from arithmetic import reconstruct
P=Path(__file__).resolve().parent;OUT=P/'output'
def read(path):return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify(pins):
    assert type(pins)is dict and pins
    for path,h in pins.items():assert type(h)is str and re.fullmatch('[0-9a-f]{64}',h) and sha(path)==h,path

def decode(x):
    if type(x)is dict:
        if 'q'in x:
            assert set(x)=={'q'} and type(x['q'])is list and len(x['q'])==2
            a,b=x['q'];assert type(a)is type(b)is str
            n,d=int(a),int(b);assert str(n)==a and str(d)==b and d>0 and math.gcd(n,d)==1
            return F(n,d)
        return {k:decode(v) for k,v in x.items()}
    if type(x)is list:return [decode(v) for v in x]
    assert type(x) in (str,int,bool,type(None)),'unexpected scalar type'
    return x

def encode(x):
    if type(x)is F:return {'q':[str(x.numerator),str(x.denominator)]}
    if type(x)is dict:return {k:encode(v) for k,v in x.items()}
    if type(x)is list:return [encode(v) for v in x]
    return x

def equal(a,b):
    assert type(a)is type(b),(type(a),type(b))
    if type(a)is dict:
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a:equal(a[k],b[k])
    elif type(a)is list:
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True):equal(x,y)
    else:assert a==b,(a,b)

def main():
    assert not OUT.exists(),'existing immutable review output'
    OUT.mkdir();start=time.process_time()
    result={'complete':False,'pass_all':False,'rows':[],'event_count':0,'systems':0,'nonsingular':0,'singular':0,'candidate_calls':0,'trained_calls':0}
    def save():
        result['cpu_seconds']=time.process_time()-start;result['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp=OUT/'results.tmp';tmp.write_text(json.dumps(encode(result),indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'results.json')
    def event(kind,**kw):
        save();assert time.process_time()-start<17,'internal review cap'
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'kind':kind,'cpu_seconds':time.process_time()-start,**kw})+'\n')
        result['event_count']+=1;save()
    try:
        save();p=read(P/'protocol.json');g=read(P/'gate.json');done=read(P/'completed_inputs.json')
        assert g['ready'] is True
        for key in ['source_review_commit','source_review_publication','execution_scope_publication']:assert re.fullmatch('[0-9a-f]{8,40}',g[key])
        verify(p['pins']);verify(g['pins']);verify(done['pins'])
        assert re.fullmatch('[0-9a-f]{8,40}',done['producer_publication'])
        own=[str(P/f) for f in ['check.py','arithmetic.py','fixtures.py','protocol.json','PROTOCOL.md','launch.sh']]
        assert all(f in g['pins'] for f in own+[str(P/'completed_inputs.json'),p['source_review']])
        sr=read(p['source_review']);assert all(sr[k] is True for k in ['complete','pass_all','execution_authorized']);verify(sr['pins']);assert all(f in sr['pins'] for f in own)
        assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert sys.executable==p['invoked_python'] and str(Path(sys.executable).resolve())==p['resolved_python'] and sha(sys.executable)==p['python_sha256']
        assert all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'])
        root=Path(p['producer']);pp=read(root/'protocol.json');pg=read(root/'gate.json')
        verify(pp['pins']);verify(pg['pins']);prior=read(pp['prior_written_review']);assert prior['complete'] is prior['pass_all'] is True;verify(prior['pins'])
        for key in ['source_review_publication','execution_scope_publication']:assert re.fullmatch('[0-9a-f]{8,40}',pg[key])
        pr=read(pg['source_review']);assert all(pr[k] is True for k in ['complete','pass_all','execution_authorized']);verify(pr['pins'])
        assert pg['source_review'] in pg['pins'] and pg['source_review'] in done['pins']
        for f in ['algebra.py','fixtures.py','run.py','PROTOCOL.md','protocol.json','launch.sh']:
            assert str(root/f) in pr['pins'] and str(root/f) in pg['pins']
        required=[str(root/f) for f in ['output/results.json','output/progress.jsonl','run1.log','gate.json']]
        assert all(f in done['pins'] for f in required)
        raw=read(root/'output/results.json');assert raw['complete'] is raw['pass_all'] is True
        assert type(raw['cpu_seconds']) in (int,float) and math.isfinite(raw['cpu_seconds']) and raw['cpu_seconds']>=0
        assert raw['protocol_sha256']==sha(root/'protocol.json') and raw['gate_sha256']==sha(root/'gate.json')
        assert raw['runtime']==dict(invoked=pp['invoked_python'],resolved=pp['resolved_python'],version=sys.version,cpu_limits=[16,20],address_space_limits=[2**31,2**31])
        assert pp['invoked_python']==sys.executable and pp['resolved_python']==p['resolved_python'] and pp['python_sha256']==p['python_sha256']
        assert raw['solve_attempts']==36 and raw['solve_returns']==35 and len(raw['rows'])==17 and raw['event_count']==36
        result.update(pins={**p['pins'],**done['pins'],str(P/'gate.json'):sha(P/'gate.json')},runtime={'invoked':sys.executable,'resolved':p['resolved_python'],'version':sys.version,'cpu_limits':[19,20],'address_space_limits':[2**31,2**31]},producer_cpu=raw['cpu_seconds'])
        events=[json.loads(line) for line in (root/'output/progress.jsonl').read_text().splitlines()]
        expected_kinds=['started']+sum((['case_started','case_passed'] for _ in range(17)),[])+['completed']
        assert len(events)==36 and [e['kind'] for e in events]==expected_kinds
        times=[e['cpu_seconds'] for e in events];assert all(type(t) in (int,float) and math.isfinite(t) and t>=0 for t in times) and times==sorted(times) and times[-1]<=raw['cpu_seconds']<16
        assert events[0]['protocol_sha256']==raw['protocol_sha256'] and events[0]['gate_sha256']==raw['gate_sha256']
        assert {k:events[-1][k] for k in ['cases','solve_attempts','solve_returns']}==dict(cases=17,solve_attempts=36,solve_returns=35)
        assert type(raw['max_rss_kib']) is int and 0<raw['max_rss_kib']<2097152
        event('started')
        cases=inventory();assert len(cases)==17
        for ix,(f,stored) in enumerate(zip(cases,raw['rows'],strict=True)):
            event('case_started',index=ix,name=f['name'])
            equal(f,stored['fixture']);assert stored['index']==ix and stored['name']==f['name'] and stored['status']=='passed'
            expected,dets=reconstruct(f,ix)
            rec={'index':ix,'name':f['name'],'determinants':dets,'expected':expected,'full_saved_row_equal':False}
            result['rows'].append(rec);save()  # Return is durable before assertions.
            equal(expected,decode(stored));rec['full_saved_row_equal']=True
            result['systems']+=len(dets);result['nonsingular']+=sum(d!=0 for d in dets);result['singular']+=sum(d==0 for d in dets)
            for e in events[1+2*ix:3+2*ix]:assert e['index']==ix and e['name']==f['name']
            save();event('case_checked',index=ix,name=f['name'])
        assert result['systems']==36 and result['nonsingular']==35 and result['singular']==1
        result.update(cases=17,producer_events=36,qualification='Fixed exact synthetic identity/exception checks only; no candidate replay, trained performance, finite-program parity, loss-family or 99% claim.')
        event('completed');assert result['event_count']==36
        result['complete']=result['pass_all']=True;save()
    except BaseException:
        result.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
