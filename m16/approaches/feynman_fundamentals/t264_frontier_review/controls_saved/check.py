"""Independent saved-only rational certificate checks; never imports producer."""
import hashlib, json, resource, time, traceback
from fractions import Fraction as F
from pathlib import Path
P = Path(__file__).resolve().parent

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(v):
    assert isinstance(v, list) and len(v) == 2 and all(isinstance(x,str) for x in v)
    a,b=map(int,v); assert str(a)==v[0] and str(b)==v[1] and b>0
    z=F(a,b); assert z.numerator==a and z.denominator==b
    return z
def enc(v):
    if isinstance(v,F): return [str(v.numerator),str(v.denominator)]
    if isinstance(v,dict): return {k:enc(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [enc(x) for x in v]
    return v

def inspect(case,row):
    assert row['name']==case['name'] and row['status']=='passed'
    r=row['raw']; e=list(map(F,case['e'])); c=list(map(F,case['c'])); O=F(case['O']); Q=F(case['Q'])
    M=sum(c,F()); assert O>M and Q>0 and min(e+c+[F()])>=0
    for key, vals in [('errors',e),('costs',c)]: assert list(map(q,r[key]))==vals
    for key,z in [('total',O),('threshold',Q),('fixed_cost',O-M),('removable',M)]: assert q(r[key])==z
    ids=r['order']; assert sorted(ids)==[i for i,z in enumerate(c) if z>0]
    assert all((e[a]*c[b],a)<=(e[b]*c[a],b) for a,b in zip(ids,ids[1:]))
    # Direct sums for each saved prefix, independently of the producer's incremental loop.
    S=[sum((c[i] for i in ids[:j]),F()) for j in range(len(ids)+1)]
    E=[sum((e[i] for i in ids[:j]),F()) for j in range(len(ids)+1)]
    flags=[b*b*(O-s)<=Q for s,b in zip(S,E)]
    assert len(r['breakpoints'])==len(S)
    for z,s,b,ok in zip(r['breakpoints'],S,E,flags):
        assert q(z['saved'])==s and q(z['error'])==b and z['feasible'] is ok
    j=r['last_feasible_breakpoint']; assert type(j) is int and flags[j] and not any(flags[j+1:])
    opt=F(case['optimum']); lo=q(r['fractional_saved_lower']); hi=q(r['fractional_saved_upper'])
    assert 0<=lo<=opt<=hi<=M and hi-lo<=M/F(2**80)
    assert q(r['fractional_cost_lower'])==2*(O-hi)
    trace=r['trace']
    if j==len(ids):
        assert lo==hi==opt==M and not trace and r['boundary'] is None
    else:
        i=ids[j]; a=e[i]/c[i]; z=r['boundary']
        assert z['index']==i and q(z['base_saved'])==S[j] and q(z['base_error'])==E[j] and q(z['slope'])==a
        assert S[j]<=opt<S[j+1] and len(trace)==80
        err=E[j]+a*(opt-S[j]); assert err*err*(O-opt)==Q
        assert err>0 and 2*a*(O-opt)-err>=0
        # Check each stored decision using expanded cubic coefficients, not the original product expression.
        b=E[j]-a*S[j]; coeff=(-a*a, a*a*O-2*a*b, 2*a*b*O-b*b, b*b*O-Q)
        left,right=S[j],S[j+1]
        for t in trace:
            mid=q(t['mid']); assert 2*mid==left+right
            assert q(t['error'])==a*mid+b
            polynomial=((coeff[0]*mid+coeff[1])*mid+coeff[2])*mid+coeff[3]
            ok=polynomial<=0; assert t['feasible'] is ok
            if ok: left=mid
            else: right=mid
        assert (left,right)==(lo,hi)
    assert set(r['methods'])=={'error_fixed','error_self','ratio_self'}
    savings={}
    for name,m in r['methods'].items():
        order=ids if name=='ratio_self' else sorted(range(len(e)),key=lambda i:(e[i],i))
        candidates=[(F(),[],F())]; assert len(m['prefixes'])==len(order)
        for n,z in enumerate(m['prefixes'],1):
            subset=order[:n]; s=sum((c[i] for i in subset),F()); b=sum((e[i] for i in subset),F())
            ok=b*b*(O if name=='error_fixed' else O-s)<=Q
            assert z['index']==subset[-1] and q(z['saved'])==s and q(z['error'])==b and z['feasible'] is ok
            if ok: candidates.append((s,subset,b))
        best=max(candidates,key=lambda t:t[0]); z=m['best']
        assert q(z['saved'])==best[0] and z['indices']==best[1] and q(z['error'])==best[2]
        assert best[2]**2*(O-best[0])<=Q and best[0]<=opt
        savings[name]=best[0]
    for name,v in case.get('expected_method_saved',{}).items(): assert savings[name]==F(v)
    if 'probe' in case:
        z=case['probe']; ok=F(z['error'])**2*(O-F(z['saved']))<=Q
        assert row['probe_feasible'] is ok and ok is z['feasible']
    return {'name':case['name'],'optimum':opt,'bracket':[lo,hi],'trace_decisions':len(trace),'method_savings':savings,'breakpoints':len(S)}

def main():
    out=P/'results.json'; progress=P/'progress.jsonl'
    assert not out.exists() and not progress.exists()
    start=time.process_time(); result={'complete':False,'pass_all':False,'rows':[],'guards':[]}
    def save():
        result['cpu_seconds']=time.process_time()-start
        temp=P/'results.tmp'; temp.write_text(json.dumps(enc(result),indent=2)+'\n'); temp.replace(out)
    def event(name):
        with progress.open('a') as f: f.write(json.dumps({'event':name,'cpu_seconds':time.process_time()-start})+'\n')
        assert time.process_time()-start<17
    try:
        protocol=load(P/'protocol.json'); gate=load(P/'gate.json')
        assert gate['ready'] is True
        for key in ['review_commit','review_publication','source_publication']:
            z=gate[key]; assert isinstance(z,str) and 8<=len(z)<=40 and all(c in '0123456789abcdef' for c in z)
        required=[str(P/n) for n in ['check.py','protocol.json','PROTOCOL.md']]
        assert all(p in gate['pins'] for p in required) and gate['review'] in gate['pins']
        for pins in [protocol['pins'],gate['pins']]:
            for p,h in pins.items(): assert sha(p)==h,p
        review=load(gate['review']); assert all(review[k] is True for k in ['complete','pass_all','execution_allowed'])
        assert all(p in review['pins'] for p in required)
        for p,h in review['pins'].items(): assert sha(p)==h,p
        base=Path(protocol['producer'])
        for n in ['protocol.json','gate.json']:
            z=load(base/n)
            for p,h in z['pins'].items(): assert sha(p)==h,p
        pg=load(base/'gate.json'); sr=load(pg['source_review'])
        assert pg['ready'] is True and all(sr[k] is True for k in ['complete','pass_all','execution_allowed'])
        for p,h in sr['pins'].items(): assert sha(p)==h,p
        result['limits']={'cpu':resource.getrlimit(resource.RLIMIT_CPU),'address_space':resource.getrlimit(resource.RLIMIT_AS)}; save()
        assert result['limits']=={'cpu':(20,20),'address_space':(2**31,2**31)}
        fixtures=load(base/'fixtures.json'); raw=load(base/'results.json')
        assert len(fixtures['valid'])==12 and len(fixtures['guards'])==6 and fixtures['bits']==80
        assert raw['complete'] is True and raw['pass_all'] is True and 'failure' not in raw
        assert raw['limits']=={'cpu':[20,20],'address_space':[2**31,2**31]}
        assert 0<=raw['cpu_seconds']<17 and raw['calls_attempted']==18 and raw['calls_returned']==12
        assert len(raw['rows'])==12 and len(raw['guards'])==6
        event('begin')
        for case,row in zip(fixtures['valid'],raw['rows']):
            result['rows'].append(inspect(case,row)); save(); event(case['name'])
        for case,row in zip(fixtures['guards'],raw['guards']):
            assert row['name']==case['name'] and row['rejected'] is True and row['message']
            try:
                ee=list(map(F,case['e'])); cc=list(map(F,case['c'])); oo=F(case['O']); qq=F(case['Q'])
                invalid=len(ee)!=len(cc) or any(x<0 for x in ee+cc) or oo<=sum(cc,F()) or qq<=0
            except ValueError: invalid=True
            assert invalid
            result['guards'].append({'name':case['name'],'invalid_independently':True}); save(); event(case['name'])
        logs=[json.loads(l) for l in (base/'progress.jsonl').read_text().splitlines()]
        expected=[('start','controls')]+[(kind,c['name']) for c in fixtures['valid']+fixtures['guards'] for kind in ['attempt','result']]+[('complete','controls')]
        assert [(x['event'],x['name']) for x in logs]==expected and len(logs)==38
        times=[x['cpu_seconds'] for x in logs]; assert all(0<=a<=b for a,b in zip(times,times[1:])) and times[-1]<=raw['cpu_seconds']
        result.update(producer_cpu_seconds=raw['cpu_seconds'],producer_events=len(logs),attempts=18,returns=12,producer_calls=0,target_payload_reads=0,pins=protocol['pins'])
        event('complete'); assert len(progress.read_text().splitlines())==20
        result.update(complete=True,pass_all=True); save()
    except BaseException:
        result.update(complete=False,pass_all=False,failure=traceback.format_exc()); save(); raise
if __name__=='__main__': main()
