"""Saved-only exact law audit, independent raw second-moment contractions."""
import copy,hashlib,json,os,pathlib,resource,sys,time
from fractions import Fraction as F
B=pathlib.Path(__file__).resolve().parent;P=pathlib.Path('/workspace/work/recover-small_h2_56/review/residual_pairing_controls');START=time.process_time();O=B/'output'
R=dict(complete=False,pass_all=False,cases=[],labels=[],guards=[])
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save():
    R.update(cpu=time.process_time()-START,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    t=O/'results.tmp';t.write_text(json.dumps(R,indent=2)+'\n');os.replace(t,O/'results.json')
def q(a):
    assert isinstance(a,list) and len(a)==2 and all(type(x)==str and str(int(x))==x for x in a)
    assert int(a[1])>0
    return F(int(a[0]),int(a[1]))
def enc(a):return [str(a.numerator),str(a.denominator)]
def moment(weights,values):return sum((weights[i]*values[i] for i in range(len(values))),F())
def table(case):
    rows=case['rows'];x=[tuple(r['x']) for r in rows];w=[q(r['weight']) for r in rows];y=[F(r['correct']) for r in rows];s=[q(r['surrogate']) for r in rows]
    assert len(set(x))==len(x) and sum(w)==1 and all(v>0 for v in w)
    rev=[]
    for a in x:
        matches=[j for j,b in enumerate(x) if all(a[k]==b[-1-k] for k in range(len(a)))]
        assert len(matches)==1;rev.extend(matches)
    invariant=all(w[i]==w[j] for i,j in enumerate(rev));a=[v-z for v,z in zip(y,s)];b=[a[j] for j in rev];c=[(v+z)/2 for v,z in zip(a,b)]
    mean=lambda z:moment(w,z)
    mu,nu,pu=map(mean,(a,b,c));var=lambda z:mean([v*v for v in z])-mean(z)**2
    va,vb,vc=map(var,(a,b,c));cov=mean([v*z for v,z in zip(a,b)])-mu*nu
    reps=[i for i,j in enumerate(rev) if x[i]<=x[j]];orbit=sum((w[i]*a[i]+(w[j]*a[j] if i!=j else 0) for i in reps for j in [rev[i]]),F())
    scal=dict(accuracy=mean(y),surrogate_mean=mean(s),residual_mean=mu,reversed_mean=nu,paired_mean=pu,surrogate_plus_pair=mean(s)+pu,absolute_residual=mean(list(map(abs,a))),absolute_reversed=mean(list(map(abs,b))),absolute_paired=mean(list(map(abs,c))),cancellation=mean([(abs(v)+abs(z)-abs(v+z))/2 for v,z in zip(a,b)]),variance=va,variance_reversed=vb,variance_paired=vc,covariance=cov,orbit_mean=orbit,half_domain_mass=sum((wi for wi,xi in zip(w,x) if xi[0]<xi[-1]),F()),half_domain_integral=sum((w[i]*(a[i]+b[i]) for i in range(len(x)) if x[i][0]<x[i][-1]),F()),endpoint_tie_mass=sum((wi for wi,xi in zip(w,x) if xi[0]==xi[-1]),F()),fixed_point_mass=sum((w[i] for i,j in enumerate(rev) if i==j),F()))
    cs,cp,budget=map(q,(case['single_cost'],case['paired_cost'],case['budget']));m=int(budget/cs);n=int(budget/cp)
    assert m>=1 and n>=1 and (not case['require_invariant'] or invariant)
    return dict(invariant=invariant,scalars={k:enc(v) for k,v in scal.items()},reverse=rev,residuals=list(map(enc,a)),paired=list(map(enc,c)),representatives=reps,single_count=m,paired_count=n,rho=enc(cov/va) if invariant and va else None,budget_variance_ratio=enc(F(m,n)*vc/va) if va else None,table_rows=len(rows),paired_residual_lookups=len(rows),model_evaluations_performed=0,declared_costs_are_hypothetical=True)
def label(row):
    x=row['x'];l=row['logits']
    def second(v):
        ranks=[sum(v[j]>v[i] or (v[j]==v[i] and j<i) for j in range(len(v))) for i in range(len(v))]
        assert ranks.count(1)==1;return ranks.index(1)
    a=second(x);b=second(list(reversed(x)));winner=next(i for i,v in enumerate(l) if all(v>=z for z in l))
    return dict(truth=a,reversed_truth=b,mapped_truth=len(x)-1-a,relabeling_holds=b==len(x)-1-a,strict_ranks=len(set(x))==len(x),first_index_winner=winner)
def malformed(f,name):
    a=copy.deepcopy(f['cases'][0])
    changes={'unnormalized_weights':('weight',['1','3']),'zero_weight':('weight',['0','1']),'negative_weight':('weight',['-1','2']),'surrogate_above_one':('surrogate',['3','2']),'nonbinary_correct':('correct',2),'boolean_correct':('correct',True),'zero_denominator':('weight',['1','0'])}
    if name in changes:k,v=changes[name];a['rows'][0][k]=v
    elif name=='duplicate_state':a['rows'][1]['x']=a['rows'][0]['x'][:]
    elif name=='not_reversal_closed':a['rows'][1]['x']=[2,-1]
    elif name=='zero_cost':a['paired_cost']=['0','1']
    elif name=='insufficient_budget':a['budget']=['1','1']
    elif name=='strict_noninvariant':a=copy.deepcopy(f['cases'][7]);a['require_invariant']=True
    else:raise AssertionError(name)
    return a

def main():
    assert not O.exists();O.mkdir();save()
    g=read(B/'gate.json');s=read(pathlib.Path(g['source_review']));inputs=read(B/'completed_inputs.json')
    assert all(s[k] is True for k in ('complete','pass_all','execution_authorized'))
    required={str(B/x) for x in ('check.py','PROTOCOL.md','launch.sh')}
    assert required<=set(s['pins']) and required<=set(g['pins'])
    mandatory={str(P/x) for x in ('output/results.json','output/progress.jsonl','run1.log','gate.json','protocol.json','fixtures.json','primitive.py','controls.py','PROTOCOL.md','launch.sh')}
    assert mandatory<=set(inputs['pins']) and g['pins'][str(B/'completed_inputs.json')]==sha(B/'completed_inputs.json')
    assert g['pins'][g['source_review']]==sha(pathlib.Path(g['source_review']))
    for pins in (s['pins'],g['pins'],inputs['pins']):
        for p,h in pins.items():assert sha(pathlib.Path(p))==h,p
    assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
    f=read(P/'fixtures.json');d=read(P/'output/results.json');events=[json.loads(l) for l in (P/'output/progress.jsonl').read_text().splitlines()]
    assert d['complete'] and d['pass_all'] and len(events)==64
    assert len(f['cases'])==len(d['cases'])==13 and len(f['labels'])==len(d['labels'])==6 and len(f['guards'])==len(d['guards'])==12
    p=read(P/'protocol.json');assert d['runtime']['invoked']==p['python'] and d['runtime']['resolved']==p['resolved_python']
    assert d['runtime']['cpu_limits']==[20,20] and d['runtime']['as_limits']==[2**31,2**31]
    kinds=['prerequisites_passed']
    for i,(case,row) in enumerate(zip(f['cases'],d['cases'])):
        assert row['index']==i and row['name']==case['name'] and row['input']==case and row['passed'] is True
        own=table(case);R['cases'].append(dict(name=case['name'],independent=own));save();assert own==row['returned']
        for k,v in case['expected'].items():assert (own['scalars'][k] if k in own['scalars'] else own[k])==v
        kinds+=['case_attempt','case_result']
    for i,(case,row) in enumerate(zip(f['labels'],d['labels'])):
        assert row['index']==i and row['input']==case and row['name']==case['name'] and row['passed'] is True
        own=label(case);assert own==case['expected']==row['returned'];R['labels'].append(own);kinds+=['label_attempt','label_result']
    for i,(name,row) in enumerate(zip(f['guards'],d['guards'])):
        assert row['index']==i and row['name']==name and row['input']==malformed(f,name) and row['rejected'] is True and row['error']
        R['guards'].append(dict(name=name,saved_rejection=True,author_guard_reexecuted=False));kinds+=['guard_attempt','guard_result']
    kinds+=['complete'];assert [e['kind'] for e in events]==kinds
    index=1
    for category,seq in [('case',f['cases']),('label',f['labels']),('guard',f['guards'])]:
        for i,x in enumerate(seq):
            name=x if type(x)==str else x['name']
            for e in events[index:index+2]:assert e['data']['index']==i and e['data']['name']==name
            index+=2
    assert (d['analyze_attempts'],d['analyze_returns'],d['label_attempts'],d['label_returns'])==(25,13,6,6)
    expected=dict(table_rows=37,paired_residual_lookups=37,exact_table_calls=13,label_calls=6,rejected_table_calls=12,model_calls=0,gaussian_calls=0,rng_calls=0)
    assert d['work']==events[-1]['data']==expected
    assert isinstance(d['cpu_seconds'],(int,float)) and 0<=d['cpu_seconds']<20 and type(d['maxrss_kib'])==int and d['maxrss_kib']>0
    assert time.process_time()-START<17
    R.update(complete=True,pass_all=True,coverage=expected,events=64,author_cpu=d['cpu_seconds'],scope='finite given tables only; guard rejection source/saved audit; no author/Gaussian/model/RNG calls');save()
if __name__=='__main__':
    try:main()
    except BaseException as e:
        R.update(complete=False,pass_all=False,failure={'type':type(e).__name__,'message':str(e)})
        if O.exists():save()
        raise
