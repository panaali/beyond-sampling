"""Frozen retained-summary arithmetic only. Never imports owner code or NumPy."""
import hashlib, json, math, os, pathlib, resource, sys, time
from fractions import Fraction as F
P=pathlib.Path(__file__).resolve().parent
B=P.parent
OUT=P/'output'
START=time.process_time()
R={'complete':False,'pass_all':False,'records':[],'counts':{},'unknowns':[]}

def save():
    tmp=OUT/'result.tmp'
    with tmp.open('w') as f:
        json.dump(R,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,OUT/'result.json')

def event(kind, **data):
    with (OUT/'progress.jsonl').open('a') as f:
        f.write(json.dumps({'kind':kind,**data},allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    assert time.process_time()-START<17, 'internal CPU cap'

def check(name, passed, **raw):
    R['records'].append({'name':name,'passed':bool(passed),**raw});save()
    assert passed,name

def pins(d):
    for path,h in d.items():
        assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==h,path

def count(name,p,n):
    assert type(n) is int and 0<n<=1000000
    assert isinstance(p,(int,float)) and math.isfinite(p) and 0<=p<=1
    p=float(p)
    lo=(F(math.nextafter(p,-math.inf))+F(p))/2*n
    hi=(F(math.nextafter(p,math.inf))+F(p))/2*n
    a=max(0,-((-lo.numerator)//lo.denominator));b=min(n,hi.numerator//hi.denominator)
    candidates=[k for k in range(a,b+1) if float(F(k,n))==p]
    R['counts'][name]={'denominator':n,'float_hex':p.hex(),'rounding_interval_integer_range':[a,b],'compatible_counts':candidates}
    if len(candidates)!=1:R['unknowns'].append(name)
    save()
    assert len(candidates)==1,'ambiguous/incompatible '+name
    return candidates[0]

def rate(name,k,n,p):
    if n==0:
        check(name,p is None,expected_none=True);return
    check(name,float(F(k,n))==p,numerator=k,denominator=n,saved=p)

def rates(name,row,n):
    return {k:count(name+'/'+k,v,n) for k,v in row.items() if k in ('acc','agree')}

def main():
    protocol=json.loads((P/'protocol.json').read_text()); gate=json.loads((P/'gate.json').read_text())
    pins(protocol['pins']);pins(gate['pins'])
    for k in ('source_review_commit','publication'):
        assert 8<=len(gate[k])<=40 and all(c in '0123456789abcdef' for c in gate[k])
    reviewpath=gate['source_review_path'];review=json.loads(pathlib.Path(reviewpath).read_text())
    assert review['complete'] and review['pass_all'] and review['execution_allowed']
    mandatory=[str(P/'check.py'),str(P/'protocol.json'),str(P/'PROTOCOL.md'),str(B/'retained/inventory.json')]
    for path in mandatory:
        assert path in review['pins'] and path in gate['pins']
        assert review['pins'][path]==gate['pins'][path]
    assert reviewpath in gate['pins'];pins(review['pins'])
    written=json.loads(pathlib.Path(gate['written_review_path']).read_text())
    assert written['complete'] and written['pass_all'];assert gate['written_review_path'] in gate['pins']
    pins(written['pins'])
    for path in (str(B/'SOURCE_REVIEW.md'),str(B/'manifest.json')):
        assert written['pins'][path]==protocol['pins'][path]
    assert pathlib.Path(sys.executable).resolve()==pathlib.Path('/usr/bin/python3').resolve()
    cpu=resource.getrlimit(resource.RLIMIT_CPU);mem=resource.getrlimit(resource.RLIMIT_AS)
    R['runtime']={'executable':sys.executable,'version':sys.version,'cpu':cpu,'memory':mem}
    save();assert all(0<x<=20 for x in cpu) and all(0<x<=2*1024**3 for x in mem)
    event('begin')
    inv=json.loads((B/'retained/inventory.json').read_text())
    for r in inv['records']:
        if r['present']:assert hashlib.sha256((B/'retained'/r['snapshot']).read_bytes()).hexdigest()==r['sha256']
    C=json.loads((B/'retained/results/phaseC2.json').read_text())
    V=json.loads((B/'retained/results/phaseC.json').read_text())
    D=json.loads((B/'retained/results/phaseD.json').read_text())
    E=json.loads((B/'retained/results/phaseC_diag.json').read_text())
    check('sample_sizes',C['N']==V['N']==[1000000,200000,200000] and D['N']==200000)
    n=D['N'];names=C['families'];check('families',len(names)==12 and len(set(names))==12)
    for field in ('val_r2','eval_r2'):
        ar=C[field];check(field+'/shape',len(ar)==12 and all(len(t)==11 and all(len(v)==16 and all(math.isfinite(x) for x in v) for v in t) for t in ar))
    selected=[[names[max(range(12),key=lambda f:C['val_r2'][f][t][j])] for j in range(16)] for t in range(1,11)]
    check('validation_selection',selected==C['best_family'],reconstructed=selected)
    for choice in ('best','best_noH'):
        rows=C['cumulative'][choice]
        check(choice+'/groups',len(rows)==6 and rows[-1]['S']==[4,2,5,9,1,6,7,8,3,13,0,10,11,12,14,15])
        check(choice+'/final_identity',all(rows[-1][k]==v for k,v in C['cumulative'][choice+'_final_step_only'].items()))
        for j,row in enumerate(rows):rates(choice+'/cumulative/'+str(j),row,n)
        for kind in ('single','group'):
            for name,row in C[kind][choice].items():rates(choice+'/'+kind+'/'+name,row,n)
    order=sorted(range(16),key=lambda j:-C['single']['best']['n'+str(j)]['agree'])
    check('evaluation_selected_order',[r['added'] for r in C['cumulative_by_neuron']]==['n'+str(j) for j in order],order=order)
    check('per_neuron_endpoint',all(C['cumulative_by_neuron'][-1][k]==v for k,v in C['cumulative']['best_final_step_only'].items()))
    event('selection_complete')
    conf={k:count('confusion/'+k,v,n) for k,v in D['confusion'].items()}
    keys=['net_ok&rep_ok','net_ok&rep_wrong','net_wrong&rep_ok','net_wrong&rep_wrong_same','net_wrong&rep_wrong_diff']
    check('confusion_keys',list(conf)==keys)
    a,b,c,d,e=[conf[k] for k in keys];dis=b+c+e;net=a+b;rep=a+c
    check('confusion_partition',sum(conf.values())==n)
    rate('disagreement',dis,n,D['disagree']);rate('C2_base',net,n,C['base']['acc'])
    end=C['cumulative']['best'][-1]
    rate('C2_accuracy',rep,n,end['acc']);rate('C2_agreement',n-dis,n,end['agree'])
    rate('error_coverage',c+e,n-net,D['disagree_given']['net_wrong']);rate('correct_disagreement',b,net,D['disagree_given']['net_correct'])
    R['paired']={'gain_count':c-b,'gain_exact':str(F(c-b,n)),'population_variance_exact':str(F(b+c,n)-F(c-b,n)**2),'mean_sample_variance_exact':str((F(b+c,n)-F(c-b,n)**2)/(n-1))};save()
    casecounts={}
    for name,row in D['by_case'].items():
        support=count('case/'+name+'/support',row['share'],n);hits=count('case/'+name+'/hits',row['disagree'],support)
        rate('case/'+name+'/share_dis',hits,dis,row['share_of_disagreements']);casecounts[name]=(support,hits)
    check('exclusive_cases',sum(casecounts[k][0] for k in ('x9max','x9second','x9below'))==n and sum(casecounts[k][1] for k in ('x9max','x9second','x9below'))==dis)
    near=D['near_tie_or_blind'];ns=count('near/support',near['share'],n);nh=count('near/hits',near['disagree'],ns)
    rate('near/share_dis',nh,dis,near['share_of_disagreements']);rate('far/dis',dis-nh,n-ns,near['disagree_elsewhere'])
    totals={}
    for name in ('by_gap23','by_gap12','by_full_gate_pattern_count','by_full_gate_pattern_count_near','by_full_gate_pattern_count_far'):
        accum=[0,0,0];raw=[]
        for i,row in enumerate(D[name]):
            tag=name+'/'+str(i);support=count(tag+'/support',row['share'],n)
            if support:
                hits=count(tag+'/hits',row['disagree'],support);errors=count(tag+'/errors',row['net_err'],support)
            else:
                check(tag+'/empty',row['disagree'] is None and row['net_err'] is None);hits=errors=0
            if 'share_of_disagreements' in row:rate(tag+'/share_dis',hits,dis,row['share_of_disagreements'])
            accum=[accum[0]+support,accum[1]+hits,accum[2]+errors];raw.append([support,hits,errors])
        expected=(ns,nh) if name.endswith('_near') else ((n-ns,dis-nh) if name.endswith('_far') else (n,dis))
        check(name+'/partition',accum[:2]==list(expected),counts=raw,totals=accum);totals[name]=accum
        if not name.endswith(('_near','_far')):check(name+'/errors',accum[2]==n-net)
    check('near_far_errors',totals['by_full_gate_pattern_count_near'][2]+totals['by_full_gate_pattern_count_far'][2]==n-net)
    fe=totals['by_full_gate_pattern_count_far'][2];far=D['far_disagree_given']
    rate('far/error_rate',fe,n-ns,far['net_err_far'])
    bad=count('far/error_disagree',far['net_wrong'],fe);good=count('far/correct_disagree',far['net_correct'],n-ns-fe)
    check('far/disjoint_errors',bad+good==dis-nh)
    check('rank_histogram_keys',set(D['rank_picked_when_disagree'])=={'replacement','network'})
    for name,counts in D['rank_picked_when_disagree'].items():check('rank/'+name,len(counts)==10 and all(type(x)is int and x>=0 for x in counts) and sum(counts)==dis,counts=counts)
    event('partition_complete')
    check('noise_rows',len(E['noise'])==7 and [r['1-R2'] for r in E['noise']]==[1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5])
    for i,row in enumerate(E['noise']):rates('noise/'+str(i),row,200000)
    for name,row in E.items():
        if name!='noise':
            for k,v in row.items():
                if k.endswith(('_agree','_acc')):count('direct/'+name+'/'+k,v,200000)
    # Logs supply text provenance only; no parser imports/evaluation.
    log=(B/'retained/logs/phaseC2.log').read_text().splitlines()
    check('fit_log_steps',[x for x in log if x.startswith('fit t ')]==['fit t '+str(i) for i in range(1,11)])
    R['scope']='retained arithmetic conditional on correctly rounded mean division; not raw prediction or new-seed reproduction'
    R['cpu_seconds']=time.process_time()-START
    event('complete');R['complete']=R['pass_all']=True;save()

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'result.json').exists() and not (OUT/'progress.jsonl').exists()
    try:main()
    except BaseException as exc:
        R['complete']=R['pass_all']=False;R['fatal']={'type':type(exc).__name__,'message':str(exc),'cpu_seconds':time.process_time()-START};save()
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'kind':'fatal',**R['fatal']})+'\n')
        raise
