"""Independent retained-byte frontier audit. Never imports producer code."""
import ast,gzip,hashlib,json,math,resource,sys,time,traceback,zipfile
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(pins):
    for p,h in pins.items():assert sha(p)==h,p
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def q(v):
    assert isinstance(v,list) and len(v)==2 and all(isinstance(z,str) for z in v)
    a,b=map(int,v);assert str(a)==v[0] and str(b)==v[1] and b>0
    x=F(a,b);assert x.numerator==a and x.denominator==b
    return x
def enc(v):
    if isinstance(v,F):return [str(v.numerator),str(v.denominator)]
    if isinstance(v,dict):return {k:enc(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [enc(x) for x in v]
    return v
def exactint(x):
    assert isinstance(x,(int,float,F)) and not isinstance(x,bool)
    y=F(x);assert y.denominator==1 and 0<=y<=2**53
    return int(y)
def binary64(raw):
    u=int.from_bytes(raw,'little');power=(u>>52)&2047;mantissa=u&((1<<52)-1)
    assert power!=2047
    if power:mantissa+=1<<52;exponent=power-1075
    else:exponent=-1074
    if u>>63:mantissa=-mantissa
    return F(mantissa*(1<<max(exponent,0)),1<<max(-exponent,0))
def readarrays(path):
    keys={'err','mass_ops','simplices','kind','leaf_mass','rays'};out={};headers={}
    with zipfile.ZipFile(path) as archive:
        assert len(archive.infolist())==6 and set(archive.namelist())=={k+'.npy' for k in keys}
        for k in sorted(keys):
            info=archive.getinfo(k+'.npy');assert info.file_size<=865536
            data=archive.read(info);assert data[:6]==b'\x93NUMPY'
            ver=tuple(data[6:8]);assert ver in [(1,0),(2,0)]
            w={1:2,2:4}[ver[0]];hs=int.from_bytes(data[8:8+w],'little');assert 0<hs<=65536
            start=8+w+hs;h=ast.literal_eval(data[8+w:start].decode('latin1'))
            assert set(h)=={'descr','fortran_order','shape'} and h['descr']=='<f8' and h['fortran_order'] is False
            assert isinstance(h['shape'],tuple) and len(h['shape'])==1 and type(h['shape'][0]) is int
            n=h['shape'][0];assert 0<n<=100000 and len(data)==start+8*n
            out[k]=[binary64(data[i:i+8]) for i in range(start,len(data),8)]
            headers[k]={'version':ver,'header':h,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    assert len({len(v) for v in out.values()})==1
    return out,headers

def envelope(raw,e,c,O,Q):
    n=len(e);M=sum(c)
    assert list(map(q,raw['errors']))==e and list(map(q,raw['costs']))==c
    for k,v in [('total',O),('threshold',Q),('removable',M),('fixed_cost',O-M)]:assert q(raw[k])==v
    ids=raw['order'];assert all(type(i) is int for i in ids)
    assert len(ids)==len(set(ids)) and set(ids)=={i for i,x in enumerate(c) if x>0}
    assert all((e[a]*c[b],a)<=(e[b]*c[a],b) for a,b in zip(ids,ids[1:]))
    points=raw['breakpoints'];assert len(points)==len(ids)+1
    s=b=F();last=0;pointvalues=[]
    for j,z in enumerate(points):
        if j:s+=c[ids[j-1]];b+=e[ids[j-1]]
        ok=b*b*(O-s)<=Q
        assert q(z['saved'])==s and q(z['error'])==b and z['feasible'] is ok
        pointvalues.append((s,b))
        if ok:last=j
    assert raw['last_feasible_breakpoint']==last
    lo=q(raw['fractional_saved_lower']);hi=q(raw['fractional_saved_upper'])
    assert 0<=lo<=hi<=M and hi-lo<=F(M,2**80)
    if last==len(ids):assert lo==hi==M and raw['trace']==[] and raw['boundary'] is None
    else:
        left,base=pointvalues[last];right=pointvalues[last+1][0];a=e[ids[last]]/c[ids[last]];intercept=base-a*left
        z=raw['boundary'];assert z['index']==ids[last] and q(z['base_saved'])==left and q(z['base_error'])==base and q(z['slope'])==a
        coefficients=(-a*a,a*a*O-2*a*intercept,2*a*intercept*O-intercept**2,intercept**2*O-Q)
        assert len(raw['trace'])==80
        for t in raw['trace']:
            x=q(t['mid']);assert 2*x==left+right and q(t['error'])==a*x+intercept
            val=((coefficients[0]*x+coefficients[1])*x+coefficients[2])*x+coefficients[3]
            ok=val<=0;assert t['feasible'] is ok
            if ok:left=x
            else:right=x
        assert (left,right)==(lo,hi)
    assert q(raw['fractional_cost_lower'])==2*(O-hi)
    assert set(raw['methods'])=={'error_fixed','error_self','ratio_self'}
    methods={}
    for name,m in raw['methods'].items():
        order=[z['index'] for z in m['prefixes']];assert all(type(i) is int for i in order)
        if name=='ratio_self':assert order==ids
        else:
            assert len(order)==n and set(order)==set(range(n))
            assert all((e[a],a)<=(e[b],b) for a,b in zip(order,order[1:]))
        s=b=F();best_s=best_b=F();best_len=0
        for j,z in enumerate(m['prefixes'],1):
            i=order[j-1];s+=c[i];b+=e[i]
            ok=b*b*(O if name=='error_fixed' else O-s)<=Q
            assert q(z['saved'])==s and q(z['error'])==b and z['feasible'] is ok
            if ok and s>best_s:best_s,best_b,best_len=s,b,j
        best=m['best'];selected=best['indices'];assert selected==order[:best_len]
        assert q(best['saved'])==best_s==sum((c[i] for i in selected),F())
        assert q(best['error'])==best_b==sum((e[i] for i in selected),F())
        assert best_b**2*(O-best_s)<=Q and best_s<=hi
        methods[name]={'saved':best_s,'error':best_b,'count':best_len}
    assert methods['ratio_self']['saved']>=pointvalues[last][0]
    assert hi-methods['ratio_self']['saved']<=max(c,default=0)
    return methods,lo,hi,len(ids),len(raw['trace'])

def main():
    out=P/'results.json';progress=P/'progress.jsonl';assert not out.exists() and not progress.exists()
    start=time.process_time();r={'complete':False,'pass_all':False}
    def save():
        r['cpu_seconds']=time.process_time()-start;tmp=P/'results.tmp';tmp.write_text(json.dumps(enc(r),indent=2,allow_nan=False)+'\n');tmp.replace(out)
    def event(name):
        with progress.open('a') as f:f.write(json.dumps({'event':name,'cpu_seconds':time.process_time()-start})+'\n')
        assert time.process_time()-start<17
    try:
        p=load(P/'protocol.json');g=load(P/'gate.json');verify(p['pins']);verify(g['pins']);assert g['ready'] is True
        for k in ['review_commit','review_publication','source_publication']:
            z=g[k];assert isinstance(z,str) and 8<=len(z)<=40 and all(c in '0123456789abcdef' for c in z)
        assert g['review'] in g['pins'];review=load(g['review']);verify(review['pins'])
        assert all(review[k] is True for k in ['complete','pass_all','execution_allowed'])
        for name in ['check.py','protocol.json','PROTOCOL.md']:assert str(P/name) in g['pins'] and str(P/name) in review['pins']
        B=Path(p['producer']);pp=load(B/'target_protocol.json');pg=load(B/'target_gate.json');verify(pp['pins']);verify(pg['pins'])
        assert pg['ready'] is True
        for name in ['source_review','parity_review']:
            rv=load(pg[name]);verify(rv['pins']);assert rv['complete'] is True and rv['pass_all'] is True
            if name=='source_review':assert rv['execution_allowed'] is True
        r['limits']={'cpu':list(resource.getrlimit(resource.RLIMIT_CPU)),'address_space':list(resource.getrlimit(resource.RLIMIT_AS))};save()
        assert r['limits']=={'cpu':[20,20],'address_space':[2**31,2**31]}
        assert str(Path(sys.executable).resolve())==p['runtime'] and p['runtime'] in p['pins']
        event('begin')
        manifest=load(B/'retained/manifest.json');assert len(manifest['records'])==3
        for item in manifest['records']:
            data=Path(item['staged_path']).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['blob']
        arrays,headers=readarrays(B/'retained/H16_T4_zoo0_chamber.npz')
        meta=load(B/'retained/H16_T4_zoo0_chamber.json');historical=load(B/'retained/anatomy_summary.json')['H16_T4_zoo0_chamber']
        producer=load(B/'target_results.json');assert producer['complete'] is True and producer['pass_all'] is True and 'failure' not in producer
        assert producer['limits']==r['limits'] and 0<=producer['cpu_seconds']<17
        assert producer['frontier_attempts']==producer['frontier_returns']==1 and producer['target_mass_calls']==producer['model_calls']==0
        assert producer['runtime']['resolved']==p['runtime']
        n=len(arrays['err']);assert producer['decoded']['rows']==n
        assert canonical(enc(headers))==canonical(producer['decoded']['headers'])
        assert canonical(meta)==canonical(producer['decoded']['json']) and canonical(historical)==canonical(producer['decoded']['historical_summary'])
        counts={k:{'nonfinite':0,'negative':sum(v<0 for v in vals)} for k,vals in arrays.items()};assert counts==producer['input_counts']
        assert counts['err']['negative']==counts['mass_ops']['negative']==0
        e=arrays['err'];c=list(map(exactint,arrays['mass_ops']))
        for name in ['simplices','rays']:assert all(exactint(x)==x for x in arrays[name])
        assert all(x in (0,1,2) for x in arrays['kind'])
        O=exactint(meta['ops']);M=sum(c);assert M==exactint(meta['mass_ops']) and O-M==exactint(meta['geometry_ops'])>0 and n==exactint(meta['leaves'])
        for k in ['ops','mass_ops','geometry_ops','leaves','acc','err_total']:assert historical[k]==meta[k]
        prob=F(meta['acc']);assert 0<prob<1;Q=prob*(1-prob)*1189;esum=sum(e,F())
        validation={'positive_fixed_cost':O-M,'mass_cost_sum':M,'zero_cost_rows':sum(x==0 for x in c),'zero_cost_positive_error_rows':sum(x==0 and y>0 for x,y in zip(c,e)),'exact_error_sum':esum,'error_sum_minus_stored_total':esum-F(meta['err_total']),'accuracy_minus_one_minus_error_sum':prob-(1-esum),'sampler_cost':2378,'threshold_Q':Q}
        r.update(rows=n,validation=validation);save();assert canonical(enc(validation))==canonical(producer['validation']);event('decoded_and_validated')
        rawpath=B/'target_frontier.json.gz';assert sha(rawpath)==producer['raw_frontier_sha256']
        with gzip.open(rawpath,'rt',encoding='utf8') as f:raw=json.load(f)
        methods,lo,hi,positive,decisions=envelope(raw,e,c,O,Q)
        r.update(methods=methods,fractional_saved_lower=lo,fractional_saved_upper=hi,positive_cost_rows=positive,trace_decisions=decisions);save();event('frontier_checked')
        summary=producer['summary'];assert set(summary['methods'])==set(methods)
        for name,z in methods.items():
            t=summary['methods'][name];s=z['saved'];b=z['error'];assert t['selected_count']==z['count']
            vals={'saved_ops':s,'remaining_nominal_flops':2*(O-s),'remaining_fraction':(O-s)/O,'bias':b,'bias_squared_over_nominal_sampling_variance':b*b*(O-s)/Q}
            for k,v in vals.items():assert q(t[k])==v
            assert t['bias_float_diagnostic']==float(b)
            sd=t['nominal_sampling_sd_float_diagnostic'];assert isinstance(sd,float) and math.isfinite(sd) and sd>0
            assert abs(F(sd)**2-Q/(O-s))<=8*F(math.ulp(sd))*F(sd)
        feasible=max(z['saved'] for z in methods.values())
        for k,v in {'fractional_saved_lower':lo,'fractional_saved_upper':hi,'best_integer_feasible_saved':feasible,'integer_saved_gap_upper':hi-feasible}.items():assert q(summary[k])==v
        assert list(map(q,summary['integer_remaining_flop_bracket']))==[2*(O-hi),2*(O-feasible)]
        assert summary['original_nominal_flops']==2*O and summary['fixed_nominal_flops']==2*(O-M) and summary['largest_leaf_mass_ops']==max(c,default=0)
        assert canonical(summary['historical_skip_at_1se'])==canonical(historical['skip_at_1se'])
        event('summary_checked')
        events=[json.loads(x) for x in (B/'target_progress.jsonl').read_text().splitlines()]
        assert [x['event'] for x in events]==['begin','decoded','validated','frontier_attempt','frontier_return','raw_saved','complete']
        times=[x['cpu_seconds'] for x in events];assert all(0<=a<=b for a,b in zip(times,times[1:])) and times[-1]<=producer['cpu_seconds']
        r.update(producer_cpu_seconds=producer['cpu_seconds'],producer_events=7,pins=p['pins'],helper_calls=0,model_calls=0,mass_calls=0,decoded_arrays=6,retained_jsons=2)
        event('complete');assert len(progress.read_text().splitlines())==5
        r.update(complete=True,pass_all=True);save()
    except BaseException:
        r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
