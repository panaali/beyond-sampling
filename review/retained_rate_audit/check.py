"""Independent saved scalar audit. No producer imports, arrays, fit/scipy calls or RNG."""
import collections, hashlib, json, math, os, pathlib, resource, sys, time
from fractions import Fraction as F
B=pathlib.Path(__file__).resolve().parent
START=time.process_time();OWNED=False;mp=None
R={'complete':False,'pass_all':False,'comparisons':[],'groups':{},'limitations':['finite regression CIs, not asymptotic rate evidence','reference and cost provenance source qualified','no model/array/RNG/producer replay']}
OUT=B/'output'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def save():
    R['review_cpu']=time.process_time()-START
    R['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tmp=OUT/'results.tmp';tmp.write_text(json.dumps(R,indent=2,allow_nan=False)+'\n');os.replace(tmp,OUT/'results.json')
def limit():assert time.process_time()-START<17,'internal CPU cap'
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def m(v):
    if isinstance(v,F):return mp.mpf(v.numerator)/v.denominator
    if isinstance(v,float):return m(F.from_float(v))
    return mp.mpf(v)
def mean(v):return m(sum((F(x) for x in v),F()))/len(v)
def cv(x):
    if isinstance(x,dict):return {str(k):cv(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [cv(v) for v in x]
    if isinstance(x,mp.mpf):return float(x)
    return x

def compare(name,a,b,fit=False):
    if isinstance(a,dict):
        assert set(a)==set(b),(name,'keys',set(a),set(b))
        for k in a:compare(name+'.'+k,a[k],b[k],fit or k in ('alpha','ci'))
    elif isinstance(a,list):
        assert len(a)==len(b),(name,'length')
        for i,(x,y) in enumerate(zip(a,b)):compare(name+f'[{i}]',x,y,fit)
    elif a is None or isinstance(a,(str,bool)):
        assert a==b,(name,a,b)
    else:
        assert finite(a) and (finite(b) or isinstance(b,mp.mpf)),(name,'nonfinite')
        delta=abs(m(a)-m(b));tol=mp.mpf('2e-10') if fit else mp.mpf('1e-28')+mp.mpf('2e-9')*max(abs(m(a)),abs(m(b)))
        R['comparisons'].append({'field':name,'saved':a,'independent':float(b),'difference':float(delta),'tolerance':float(tol)})
        assert delta<=tol,(name,str(delta),str(tol))

TC={}
def tq(df):
    if df not in TC:
        lo=mp.mpf(0);hi=mp.mpf(100)
        for _ in range(160):
            z=(lo+hi)/2
            cdf=1-mp.betainc(mp.mpf(df)/2,mp.mpf('0.5'),0,df/(df+z*z),regularized=True)/2
            if cdf<mp.mpf('.975'):lo=z
            else:hi=z
        TC[df]=(lo+hi)/2
    return TC[df]
def slope(C,Y):
    if len(C)<3:return None
    xs=[mp.log(m(x)) for x in C];ys=[mp.log(abs(m(y))) for y in Y]
    n=len(xs);xm=sum(xs)/n;ym=sum(ys)/n
    xx=sum((x-xm)**2 for x in xs);b=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/xx
    intercept=ym-b*xm;sse=sum((y-intercept-b*x)**2 for x,y in zip(xs,ys));se=mp.sqrt(sse/(n-2)/xx)
    return {'alpha':-b,'ci':[-b-tq(n-2)*se,-b+tq(n-2)*se],'n':n}
def rows(tag):
    data=[json.loads(l) for l in (B/'frozen'/tag/'rows.jsonl').read_text().splitlines() if l]
    R[tag+'_raw']=data;save()
    for d in data:
        assert finite(d['cost']) and d['cost']>0
        assert finite(d['cpu']) and d['cpu']>=0
        assert finite(d['err'])
        assert d['est'] is None or finite(d['est'])
    return data
def identity(d,p):
    e=abs(m(d['err'])-(m(d['est'])-m(p)))
    assert e<=mp.mpf('2e-14'),('error identity',d,e)
def ranges(K,a):
    out={}
    for bar in (10.,1.):
        if abs(1-2*a)<mp.mpf('1e-9'):v='all' if K<=bar else 'none'
        elif a>.5:v=f'C >= {float((K/bar)**(1/(2*a-1))):.3g}'
        else:v=f'C <= {float((bar/K)**(1/(1-2*a))):.3g}'
        out[str(bar)]=v
    return out

MODELS=('orig','s0.005_0','s0.005_1','s0.01_0','s0.01_1');BUD=(4000,16000,64000,256000,1000000);TS=(3,4,5,6,8,10)
def t271():
    rr=rows('t271');assert len(rr)==289
    truth=load(B/'frozen/t271/dependencies/results/leaderboard_truths.json');old=load(B/'frozen/t271/fit.json')
    groups=collections.defaultdict(lambda:collections.defaultdict(list));keys=[]
    for d in rr:
        key=(d['fam'],d['model'],d['knob'],d['rep']);keys.append(key);groups[key[:2]][key[2]].append(d)
        if d['model'] in MODELS:identity(d,.9531174789 if d['model']=='orig' else truth[d['model']]['acc'])
    assert len(set(keys))==len(keys),'duplicate T271'
    wanted={(fam,n) for fam in ('sci_tao','probe_v2','probe_v4','form') for n in MODELS}|{('enum_T4','zoo8_4'),('enum_T4','zoo16_4')}
    assert set(groups)==wanted and set(old)=={f'{f}|{n}' for f,n in wanted}
    new={};ci={}
    for (fam,n),by in sorted(groups.items()):
        ks=sorted(by);is_enum=fam=='enum_T4'
        if is_enum:assert len(ks)==7 and all(type(k)==int and k>0 for k in ks)
        else:assert ks==list(BUD if fam=='sci_tao' else (1,2,4,8,16))
        for k in ks:assert sorted(d['rep'] for d in by[k])==list(range(8) if fam=='sci_tao' else range(1))
        C=[mean([d['cost'] for d in by[k]]) for k in ks];bias=[mean([d['err'] for d in by[k]]) for k in ks]
        mse=[m(sum((F(d['err'])**2 for d in by[k]),F()))/len(by[k]) for k in ks];rmse=[mp.sqrt(x) for x in mse]
        floor=2*m(3.9e-7 if n=='orig' else truth[n]['se'] if n in truth else 0);ok=[abs(b)>floor for b in bias]
        o=dict(knobs=ks,cost=C,bias=bias,rmse=rmse,span=C[-1]/C[0],reps=len(by[ks[0]]),floor=floor,fit_bias=slope(C,bias),fit_bias_resolved=slope([c for c,k in zip(C,ok) if k],[b for b,k in zip(bias,ok) if k]))
        if fam=='sci_tao':o['fit_rmse']=slope(C,rmse)
        if is_enum:
            H=int(n[3:].split('_')[0]);j=load(B/f'frozen/t271/dependencies/m16/approaches/width16_probe/anatomy/H{H}_T4_zoo0_chamber.json');p=m(j['acc']);spu=mp.mpf(2)/(4*(2*H*H+2*H)+2*H*4+4*H+12-2)
        else:p=m(.9531174789 if n=='orig' else truth[n]['acc']);spu=mp.mpf(1)
        o['ratio_measured']=[y*c*spu/(p*(1-p)) for y,c in zip(mse,C)]
        law=o.get('fit_rmse') or o['fit_bias'];ys=rmse if o.get('fit_rmse') else bias
        if law:
            a=law['alpha'];A=mp.exp(sum(mp.log(abs(y))+a*mp.log(c) for c,y in zip(C,ys))/len(C));K=A*A*spu/(p*(1-p))
            o['law']=dict(A=A,alpha=a,K=K);o['predicted_ranges']=ranges(K,a)
        key=f'{fam}|{n}';new[key]=o;R['groups']['t271:'+key]=cv(o);save();compare('t271:'+key,old[key],o);limit()
        if fam=='sci_tao':ci[n]=bool(o['fit_rmse']['ci'][0]<=.5<=o['fit_rmse']['ci'][1])
    R['t271_ci_contains_half']=ci
    R['t271_cpu_saved_row_sum']=math.fsum(d['cpu'] for d in rr)
    R['t271_cpu_note']='Rows for probe_v4 are overlapping prefix sums; sum is not distinct execution CPU.'
    R['t271_sci_ratio_extrema']={n:[float(min(new['sci_tao|'+n]['ratio_measured'])),float(max(new['sci_tao|'+n]['ratio_measured']))] for n in MODELS}
    return new

def t277():
    rr=rows('t277');assert len(rr)==720
    tr=load(B/'frozen/t277/truths.json');old=load(B/'frozen/t277/fit.json')
    assert set(tr)=={f'T{t}_s{s}' for t in TS for s in range(3)}
    exact={}
    for t in TS:
        for s in range(3):
            p=B/f'frozen/t277/dependencies/m16/approaches/width16_probe/res/H16_T{t}_zoo{s}_exact.json'
            if p.exists():exact[t,s]=load(p)['acc']
    R['t277_exact_reference_keys']=[list(k) for k in exact];R['t277_truths_raw']=tr;save()
    for k,d in tr.items():
        assert d['n']==40000000 and finite(d['cpu']) and d['cpu']>=0 and 0<d['acc']<1
        compare('truth:'+k+'.se',d['se'],mp.sqrt(m(d['acc'])*(1-m(d['acc']))/d['n']))
        assert abs(d['acc']*d['n']-round(d['acc']*d['n']))<1e-6
    groups=collections.defaultdict(list);keys=[]
    for d in rr:
        key=(d['T'],d['seed'],d['budget'],d['rep']);keys.append(key);groups[key[:3]].append(d)
        q=tr[f'T{key[0]}_s{key[1]}'];p=exact.get(key[:2],q['acc']);se=0 if key[:2] in exact else q['se']
        identity(d,p);assert d['truth_se']==se
    assert len(keys)==len(set(keys)) and set(keys)=={(t,s,b,r) for t in TS for s in range(3) for b in BUD for r in range(8)}
    new={'per_seed':{},'pooled':{}};pref={};uncorrected={};negative=[]
    for t in TS:
        pool={b:[] for b in BUD};cost={b:[] for b in BUD};ratios={b:[] for b in BUD}
        for s in range(3):
            C=[];M=[];Ra=[];Cu=[];Mu=[];p=m(tr[f'T{t}_s{s}']['acc'])
            for b in BUD:
                ds=groups[t,s,b];assert len(ds)==8
                mse=m(sum((F(d['err'])**2-F(d['truth_se'])**2 for d in ds),F()))/8
                raw=m(sum((F(d['err'])**2 for d in ds),F()))/8;c=mean([d['cost'] for d in ds])
                pool[b].append(mse);cost[b].append(c);Cu.append(c);Mu.append(mp.sqrt(raw))
                if mse>0:C.append(c);M.append(mse);Ra.append(mse*c/(p*(1-p)));ratios[b].append(Ra[-1])
                else:negative.append([t,s,b,float(mse)])
            key=f'T{t}_s{s}';uncorrected[key]=cv(slope(Cu,Mu))
            if len(C)>=3:new['per_seed'][key]=dict(fit=slope(C,[mp.sqrt(x) for x in M]),ratio=Ra,dropped=5-len(C))
        bs=[b for b in BUD if sum(pool[b])/3>0];C=[sum(cost[b])/3 for b in bs];M=[sum(pool[b])/3 for b in bs]
        assert all(len(pool[b])==3 and len(cost[b])==3 for b in BUD)
        new['pooled'][str(t)]=dict(fit=slope(C,[mp.sqrt(x) for x in M]),cost=C,mse=M)
        pref[str(t)]={str(b):float(sorted(ratios[b])[len(ratios[b])//2]) for b in BUD if len(ratios[b])==3}
        R['groups'][f't277:{t}']=cv(new['pooled'][str(t)]);save();limit()
    A=[new['pooled'][str(t)]['fit']['alpha'] for t in TS]
    ranks=[1+sum(b<a for b in A) for a in A];assert len(set(ranks))==6,'unexpected tied alpha'
    rho=1-mp.mpf(6)*sum((i+1-r)**2 for i,r in enumerate(ranks))/(6*(36-1))
    x=[mp.mpf(1)/(t-1) for t in TS];y=[a-mp.mpf('.5') for a in A];xx=sum(z*z for z in x);s=sum(z*v for z,v in zip(x,y))/xx
    se=mp.sqrt(sum((v-s*z)**2 for z,v in zip(x,y))/5/xx)
    new['summary']=dict(rho=rho,s=s,s_ci=[s-tq(5)*se,s+tq(5)*se],alphas={str(t):a for t,a in zip(TS,A)})
    R['t277_reconstructed']=cv(new);R['t277_prefactor_medians']=pref;R['t277_negative_cells']=negative;R['t277_uncorrected_diagnostic']=uncorrected;save()
    compare('t277',old,new)
    supported=A[0]>=mp.mpf('.75') and rho<=mp.mpf('-.6') and abs(A[-1]-.5)<=mp.mpf('.1') and not new['summary']['s_ci'][0]<=0<=new['summary']['s_ci'][1]
    flat=all(abs(a-.5)<=mp.mpf('.15') for a in A);refuted=flat or rho>mp.mpf('-.2')
    R['t277_finite_rule']={'supported':bool(supported),'all_within_half_plus_minus_point15':bool(flat),'rho_above_minus_point2':bool(rho>mp.mpf('-.2')),'refuted':bool(refuted),'verdict':'supported' if supported else 'refuted' if refuted else 'mixed'}
    R['t277_saved_sci_cpu_sum']=math.fsum(d['cpu'] for d in rr);R['t277_saved_truth_cpu_sum']=math.fsum(d['cpu'] for d in tr.values())
    # Preserve aggregate-only fields. No claim to derive absent enum generating rows.
    R['aggregate_only_enum']=load(B/'frozen/t277/enum.json')
    prefraw=load(B/'frozen/t277/prefactor.json');R['prefactor_raw']=prefraw
    # Numeric leaf mapping checks are recorded without assuming a missing generator's key semantics.
    leaves=[]
    def walk(v,path):
        if isinstance(v,dict):
            for k,w in v.items():walk(w,path+[k])
        elif isinstance(v,list):
            for i,w in enumerate(v):walk(w,path+[i])
        elif finite(v):leaves.append((path,v))
    walk(prefraw,[])
    R['prefactor_numeric_matches']=[]
    for path,v in leaves:
        matches=[(t,b) for t,bs in pref.items() for b,q in bs.items() if abs(v-q)<=2e-9*max(abs(v),abs(q))+1e-28]
        R['prefactor_numeric_matches'].append({'path':path,'value':v,'matching_median_keys':matches})
    R['prefactor_note']='Numeric matches do not reconstruct an absent producing script; all raw fields retained.'

def main():
    global OWNED,mp
    assert not OUT.exists(),'immutable output exists';OUT.mkdir();OWNED=True;save()
    gate=load(B/'execution_gate.json');assert gate['execution_authorized'] is True and gate['scope']=='T283 retained scalar audit'
    assert gate['pins'][gate['source_review']]==sha(pathlib.Path(gate['source_review']))
    review=load(gate['source_review']);assert all(review[k] is True for k in ('complete','pass_all','execution_authorized'))
    assert len(gate['publication'])>=8 and all(c in '0123456789abcdef' for c in gate['publication'])
    mandatory={str(B/n) for n in ('check.py','PROTOCOL.md','launch.sh','manifest.json','runtime.json')}
    assert mandatory<=set(gate['pins']) and mandatory<=set(review['pins'])
    for pins in (gate['pins'],review['pins']):
        assert pins
        for p,h in pins.items():assert sha(pathlib.Path(p))==h,(p,'pin')
    runtime=load(B/'runtime.json')
    assert sys.executable==runtime['invoked_python'] and str(pathlib.Path(sys.executable).resolve())==runtime['resolved_python'] and sys.version==runtime['python_version']
    for p,h in runtime['pins'].items():assert sha(pathlib.Path(p))==h,p
    import mpmath as numerical
    mp=numerical;assert mp.__version__==runtime['mpmath_version'] and str(pathlib.Path(mp.__file__).resolve().parent)==runtime['mpmath_root'];mp.mp.dps=60
    R['runtime']=runtime
    manifest=load(B/'manifest.json');assert manifest['complete']
    for row in manifest['records']:
        p=B/row['local'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    R['payload_count']=len(manifest['records']);R['limits']={str(k):list(resource.getrlimit(k)) for k in (resource.RLIMIT_CPU,resource.RLIMIT_AS)}
    assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2147483648,2147483648)
    t271();t277();limit();R['complete']=True;R['pass_all']=True;save()
if __name__=='__main__':
    try:main()
    except BaseException as e:
        R['complete']=False;R['pass_all']=False;R['failure']={'type':type(e).__name__,'message':str(e)}
        if OWNED:save()
        raise
