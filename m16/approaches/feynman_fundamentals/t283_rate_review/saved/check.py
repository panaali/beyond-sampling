"""Independent stdlib retained arithmetic; never imports producer/scientific modules."""
import collections,hashlib,json,math,os,resource,sys,time,traceback
from decimal import Decimal as D,getcontext
from fractions import Fraction as F
from pathlib import Path
HERE=Path(__file__).resolve().parent
SRC=Path('/workspace/work/recover-small_h2_78/review/retained_rate_audit')
OUT=HERE/'output';START=time.process_time();OWNED=False
getcontext().prec=80
R={'complete':False,'pass_all':False,'comparisons':[],'groups':{},'events':0}
FIELD={};TC={}
MODELS=['orig','s0.005_0','s0.005_1','s0.01_0','s0.01_1'];BUD=[4000,16000,64000,256000,1000000];TS=[3,4,5,6,8,10]
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dec(x):
    if isinstance(x,D):return x
    if isinstance(x,F):return D(x.numerator)/D(x.denominator)
    if type(x)is float:return D.from_float(x)
    return D(x)
def conv(x):
    if isinstance(x,(D,F)):return float(x)
    if isinstance(x,dict):return {str(k):conv(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [conv(v) for v in x]
    return x
def save():
    R['cpu_seconds']=time.process_time()-START;R['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    p=OUT/'results.tmp';p.write_text(json.dumps(conv(R),indent=2,allow_nan=False)+'\n');os.replace(p,OUT/'results.json')
def cap():assert time.process_time()-START<17,'internal CPU cap'
def event(name):
    cap();R['events']+=1
    with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':name,'cpu':time.process_time()-START})+'\n')
    save()
def finite(x):return type(x)in(int,float) and math.isfinite(x)
def moments(v):
    # All retained floats are binary64: use one integer lattice, not centered Fraction sums.
    assert v and all(finite(x) for x in v)
    q=[F(x) for x in v];den=max(x.denominator for x in q)
    assert all(den%x.denominator==0 for x in q)
    z=[x.numerator*(den//x.denominator) for x in q];n=len(z)
    return F(sum(z),n*den),F(sum(a*a for a in z),n*den*den)
def avg(v):return dec(moments(v)[0])
def compare(path,a,b,fit=False,record_field=False):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
        for k in a:compare(path+'.'+k,a[k],b[k],fit or k in ('alpha','ci'),record_field)
    elif isinstance(a,list):
        assert isinstance(b,(list,tuple)) and len(a)==len(b),(path,'length')
        for i,(x,y) in enumerate(zip(a,b)):compare(path+'['+str(i)+']',x,y,fit,record_field)
    elif a is None or type(a)in(str,bool):assert a==b,(path,a,b)
    else:
        assert finite(a) and (isinstance(b,(D,F)) or finite(b)),path
        x,y=dec(a),dec(b);tol=D('2e-10') if fit else D('1e-28')+D('2e-9')*max(abs(x),abs(y));delta=abs(x-y)
        R['comparisons'].append({'field':path,'saved':a,'computed':float(y),'difference':float(delta),'tolerance':float(tol)})
        if record_field:FIELD[path]=(a,y,fit)
        assert delta<=tol,(path,str(delta),str(tol))
def tquantile(df):
    # t=sqrt(df)*tan(theta). Its symmetric density integral is proportional
    # to integral cos(theta)^(df-1), evaluated by integer-power recurrence.
    assert type(df)is int and 1<=df<=20
    if df not in TC:
        def integral(power,theta):
            vals=[theta,math.sin(theta)]
            for j in range(2,power+1):vals.append(math.sin(theta)*math.cos(theta)**(j-1)/j+(j-1)*vals[j-2]/j)
            return vals[power]
        normal=2*integral(df-1,math.pi/2);lo,hi=0.,100.
        for _ in range(64):
            mid=(lo+hi)/2;c=.5+integral(df-1,math.atan(mid/math.sqrt(df)))/normal
            if c<.975:lo=mid
            else:hi=mid
        TC[df]=D.from_float((lo+hi)/2)
    return TC[df]
def fit(cost,y):
    if len(cost)<3:return None
    n=len(cost);x=[dec(c).ln() for c in cost];z=[abs(dec(v)).ln() for v in y]
    sx=sum(x);sz=sum(z);sxx=sum(a*a for a in x)-sx*sx/n
    sxy=sum(a*b for a,b in zip(x,z))-sx*sz/n
    assert sxx>0
    slope=sxy/sxx;intercept=(sz-slope*sx)/n
    residual=sum((b-intercept-slope*a)**2 for a,b in zip(x,z));se=(residual/D(n-2)/sxx).sqrt();q=tquantile(n-2)
    return {'alpha':-slope,'ci':[-slope-q*se,-slope+q*se],'n':n}
def crossing(K,a):
    ans={}
    for bar in (D(10),D(1)):
        if abs(1-2*a)<D('1e-9'):v='all' if K<=bar else 'none'
        elif a>D('.5'):v='C >= '+format(float(((K/bar).ln()/(2*a-1)).exp()),'.3g')
        else:v='C <= '+format(float(((bar/K).ln()/(1-2*a)).exp()),'.3g')
        ans[str(float(bar))]=v
    return ans
def rows(tag,n):
    rr=[json.loads(line) for line in (SRC/'frozen'/tag/'rows.jsonl').read_text().splitlines() if line]
    R[tag+'_raw_count']=len(rr);save();assert len(rr)==n
    for v in rr:
        assert finite(v['cost']) and v['cost']>0 and finite(v['cpu']) and v['cpu']>=0 and finite(v['err'])
        assert v['est'] is None or finite(v['est'])
    return rr
def identity(r,p):assert abs(F(r['err'])-F(r['est'])+F(p))<=F(2,10**14),('error identity',r)
def t271(author):
    rr=rows('t271',289);truth=load(SRC/'frozen/t271/dependencies/results/leaderboard_truths.json');old=load(SRC/'frozen/t271/fit.json')
    by=collections.defaultdict(list);ids=[]
    for r in rr:
        k=(r['fam'],r['model'],r['knob'],r['rep']);ids.append(k);by[k[:2]].append(r)
        if r['model'] in MODELS:identity(r,.9531174789 if r['model']=='orig' else truth[r['model']]['acc'])
    assert len(set(ids))==289
    want={(f,m) for f in ('sci_tao','probe_v2','probe_v4','form') for m in MODELS}|{('enum_T4','zoo8_4'),('enum_T4','zoo16_4')}
    assert set(by)==want and set(old)=={f+'|'+m for f,m in want}
    new={};ci={}
    for fam,model in sorted(by):
        rs=by[fam,model];knobs=sorted({r['knob'] for r in rs});enum=fam=='enum_T4';reps=8 if fam=='sci_tao' else 1
        if enum:assert len(knobs)==7 and all(type(k)is int and k>0 for k in knobs)
        else:assert knobs==(BUD if reps==8 else [1,2,4,8,16])
        assert {(r['knob'],r['rep']) for r in rs}=={(k,r) for k in knobs for r in range(reps)}
        costs=[];bias=[];mse=[]
        for k in knobs:
            xs=[r for r in rs if r['knob']==k];costs.append(avg([r['cost'] for r in xs]));mm=moments([r['err'] for r in xs]);bias.append(dec(mm[0]));mse.append(dec(mm[1]))
        rmse=[q.sqrt() for q in mse];floor=2*dec(3.9e-7 if model=='orig' else truth[model]['se'] if model in truth else 0)
        keep=[i for i,b in enumerate(bias) if abs(b)>floor]
        o={'knobs':knobs,'cost':costs,'bias':bias,'rmse':rmse,'span':costs[-1]/costs[0],'reps':reps,'floor':floor,'fit_bias':fit(costs,bias),'fit_bias_resolved':fit([costs[i] for i in keep],[bias[i] for i in keep])}
        if reps==8:o['fit_rmse']=fit(costs,rmse)
        if enum:
            h=int(model[3:].split('_')[0]);p=dec(load(SRC/f'frozen/t271/dependencies/m16/approaches/width16_probe/anatomy/H{h}_T4_zoo0_chamber.json')['acc']);spu=D(2)/D(8*h*h+20*h+10)
        else:p=dec(.9531174789 if model=='orig' else truth[model]['acc']);spu=D(1)
        assert 0<p<1
        o['ratio_measured']=[v*c*spu/(p*(1-p)) for v,c in zip(mse,costs)]
        law=o.get('fit_rmse') or o['fit_bias'];ys=rmse if o.get('fit_rmse') else bias
        if law:
            a=law['alpha'];A=(sum(abs(y).ln()+a*c.ln() for y,c in zip(ys,costs))/len(costs)).exp();K=A*A*spu/(p*(1-p))
            o.update(law={'A':A,'alpha':a,'K':K},predicted_ranges=crossing(K,a))
        key=fam+'|'+model;new[key]=o;R['groups']['t271:'+key]=o;save()
        compare('t271:'+key,old[key],o,record_field=True);compare('audit.group.t271:'+key,author['groups']['t271:'+key],o);cap()
        if reps==8:ci[model]=o['fit_rmse']['ci'][0]<=D('.5')<=o['fit_rmse']['ci'][1]
    assert ci==author['t271_ci_contains_half'];R['t271_ci_contains_half']=ci
    ex={m:[min(new['sci_tao|'+m]['ratio_measured']),max(new['sci_tao|'+m]['ratio_measured'])] for m in MODELS}
    compare('audit.t271_extrema',author['t271_sci_ratio_extrema'],ex)
    compare('audit.t271_cpu',author['t271_cpu_saved_row_sum'],sum(dec(r['cpu']) for r in rr))
    event('t271_checked')
def t277(author):
    rr=rows('t277',720);truth=load(SRC/'frozen/t277/truths.json');old=load(SRC/'frozen/t277/fit.json');exact={}
    assert set(truth)=={f'T{t}_s{s}' for t in TS for s in range(3)}
    for t in (3,4):
        for s in range(3):exact[t,s]=load(SRC/f'frozen/t277/dependencies/m16/approaches/width16_probe/res/H16_T{t}_zoo{s}_exact.json')['acc']
    assert author['t277_exact_reference_keys']==[list(k) for k in exact]
    for name,row in truth.items():
        assert row['n']==40000000 and finite(row['cpu']) and row['cpu']>=0 and 0<row['acc']<1
        p=dec(row['acc']);compare('truth:'+name+'.se',row['se'],(p*(1-p)/row['n']).sqrt(),record_field=True)
        assert abs(F(row['acc'])*row['n']-round(F(row['acc'])*row['n']))<F(1,10**6)
    by=collections.defaultdict(list);ids=[]
    for r in rr:
        k=(r['T'],r['seed'],r['budget'],r['rep']);ids.append(k);by[k[:3]].append(r)
        tr=truth[f'T{k[0]}_s{k[1]}'];identity(r,exact.get(k[:2],tr['acc']));assert r['truth_se']==(0 if k[:2] in exact else tr['se'])
    assert len(set(ids))==720 and set(ids)=={(t,s,b,r) for t in TS for s in range(3) for b in BUD for r in range(8)}
    cells={};negative=[];new={'per_seed':{},'pooled':{}};unc={};med={};alternative={}
    for t in TS:
        ratios={b:[] for b in BUD};alts={b:[] for b in BUD}
        for s in range(3):
            C=[];Y=[];Ra=[];U=[];UC=[]
            for b in BUD:
                rows_=by[t,s,b];assert len(rows_)==8
                cost=avg([r['cost'] for r in rows_]);raw=dec(moments([r['err'] for r in rows_])[1]);mse=raw-dec(moments([r['truth_se'] for r in rows_])[1]);cells[t,s,b]=(cost,mse)
                UC.append(cost);U.append(raw.sqrt())
                if mse>0:
                    p=dec(truth[f'T{t}_s{s}']['acc']);q=dec(exact.get((t,s),truth[f'T{t}_s{s}']['acc']))
                    C.append(cost);Y.append(mse.sqrt());ratio=mse*cost/(p*(1-p));Ra.append(ratio);ratios[b].append(ratio);alts[b].append(mse*cost/(q*(1-q)))
                else:negative.append([t,s,b,float(mse)])
            key=f'T{t}_s{s}';unc[key]=fit(UC,U)
            if len(C)>=3:new['per_seed'][key]={'fit':fit(C,Y),'ratio':Ra,'dropped':5-len(C)}
        C=[];M=[]
        for b in BUD:
            c=sum(cells[t,s,b][0] for s in range(3))/3;m=sum(cells[t,s,b][1] for s in range(3))/3
            if m>0:C.append(c);M.append(m)
        new['pooled'][str(t)]={'fit':fit(C,[v.sqrt() for v in M]),'cost':C,'mse':M}
        med[str(t)]={str(b):sorted(v)[1] for b,v in ratios.items() if len(v)==3}
        alternative[str(t)]={str(b):sorted(v)[1] for b,v in alts.items() if len(v)==3}
        R['groups']['t277:'+str(t)]=new['pooled'][str(t)];save();compare('audit.group.t277:'+str(t),author['groups']['t277:'+str(t)],new['pooled'][str(t)]);cap()
    A=[new['pooled'][str(t)]['fit']['alpha'] for t in TS];order=sorted(range(6),key=lambda i:A[i]);assert len(set(A))==6
    rank={j:i+1 for i,j in enumerate(order)};rho=dec(F(1)-F(6*sum((i+1-rank[i])**2 for i in range(6)),210))
    x=[D(1)/D(t-1) for t in TS];xy=sum(a*(b-D('.5')) for a,b in zip(x,A));xx=sum(a*a for a in x);s=xy/xx
    se=(sum((b-D('.5')-s*a)**2 for a,b in zip(x,A))/5/xx).sqrt();q=tquantile(5)
    new['summary']={'rho':rho,'s':s,'s_ci':[s-q*se,s+q*se],'alphas':dict(zip(map(str,TS),A))}
    R['t277_reconstructed']=new;R['t277_prefactor_mc']=med;R['t277_prefactor_exact_reference_diagnostic']=alternative;R['t277_negative_cells']=negative;R['t277_uncorrected']=unc;save()
    compare('t277',old,new,record_field=True);compare('audit.t277',author['t277_reconstructed'],new)
    compare('audit.medians',author['t277_prefactor_medians'],med);compare('audit.negative',author['t277_negative_cells'],negative);compare('audit.uncorrected',author['t277_uncorrected_diagnostic'],unc)
    supported=A[0]>=D('.75') and rho<=D('-.6') and abs(A[-1]-D('.5'))<=D('.1') and not s-q*se<=0<=s+q*se
    flat=all(abs(a-D('.5'))<=D('.15') for a in A);refuted=flat or rho>D('-.2')
    rule={'supported':supported,'all_within_half_plus_minus_point15':flat,'rho_above_minus_point2':rho>D('-.2'),'refuted':refuted,'verdict':'supported' if supported else 'refuted' if refuted else 'mixed'}
    R['finite_rule']=rule;assert rule==author['t277_finite_rule']
    compare('audit.t277_cpu',author['t277_saved_sci_cpu_sum'],sum(dec(r['cpu']) for r in rr));compare('audit.truth_cpu',author['t277_saved_truth_cpu_sum'],sum(dec(r['cpu']) for r in truth.values()))
    assert author['t277_truths_raw']==truth and author['aggregate_only_enum']==load(SRC/'frozen/t277/enum.json')
    pref=load(SRC/'frozen/t277/prefactor.json');assert pref==author['prefactor_raw'];leaves=[]
    def walk(z,path):
        if isinstance(z,dict):
            for k,v in z.items():walk(v,path+[k])
        elif isinstance(z,list):
            for i,v in enumerate(z):walk(v,path+[i])
        elif finite(z):leaves.append((path,z))
    walk(pref,[]);diagnostics=[]
    def matches(v,table):return [[t,b] for t,bs in table.items() for b,m in bs.items() if abs(dec(v)-m)<=D('1e-28')+D('2e-9')*max(abs(dec(v)),abs(m))]
    for path,v in leaves:diagnostics.append({'path':path,'value':v,'mc_matches':matches(v,med),'reference_denominator_matches':matches(v,alternative)})
    R['prefactor_diagnostic']=diagnostics;save()
    assert len(diagnostics)==len(author['prefactor_numeric_matches'])
    for a,b in zip(author['prefactor_numeric_matches'],diagnostics):assert a['path']==b['path'] and a['value']==b['value'] and a['matching_median_keys']==b['mc_matches']
    event('t277_checked')
def main():
    global OWNED
    assert not OUT.exists();OUT.mkdir();OWNED=True;save()
    p=load(HERE/'protocol.json');g=load(HERE/'gate.json');assert g['execution_authorized'] is True
    mandatory={str(HERE/n) for n in ('check.py','PROTOCOL.md','protocol.json')};assert mandatory<=set(g['pins'])
    for k in ('source_publication','review_publication'):
        v=g[k];assert isinstance(v,str) and 8<=len(v)<=40 and all(c in '0123456789abcdef' for c in v)
    assert g['pins'][g['source_review']]==sha(g['source_review']);review=load(g['source_review'])
    assert all(review[k] is True for k in ('complete','pass_all','execution_authorized')) and mandatory<=set(review['pins'])
    for pins in (p['pins'],g['pins'],review['pins']):
        for f,h in pins.items():assert sha(f)==h,f
    assert str(Path(sys.executable).resolve())==p['python_resolved']
    assert resource.getrlimit(resource.RLIMIT_CPU)==(20,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
    R['limits']={'CPU':[20,20],'AS':[2**31,2**31]};event('prerequisites')
    author=load(SRC/'output/results.json');assert author['complete'] is True and author['pass_all'] is True
    t271(author);t277(author)
    cmp=author['comparisons'];assert len(cmp)==len(FIELD)==1186 and len({r['field'] for r in cmp})==1186
    for row in cmp:
        a,b,isfit=FIELD[row['field']];assert row['saved']==a
        compare('audit.comparison.'+row['field'],row['independent'],b,isfit)
        assert 0<=row['difference']<=row['tolerance'] and finite(row['difference']) and finite(row['tolerance'])
        intended=D('2e-10') if isfit else D('1e-28')+D('2e-9')*max(abs(dec(a)),abs(b))
        assert abs(dec(row['tolerance'])-intended)<=D('1e-14')*intended
    assert set(author['groups'])==set(R['groups'])
    for tag,n in [('t271',289),('t277',720)]:assert author[tag+'_raw']==[json.loads(s) for s in (SRC/'frozen'/tag/'rows.jsonl').read_text().splitlines() if s]
    R['producer_comparisons']=len(cmp);R['student_quantiles']={str(k):str(v) for k,v in TC.items()};R['scope']='Saved finite arithmetic only; no asymptotic/equivalence or absent-generator claim.'
    event('completed');assert R['events']==4;cap();R['complete']=R['pass_all']=True;save()
if __name__=='__main__':
    try:main()
    except BaseException:
        R.update(complete=False,pass_all=False,failure=traceback.format_exc())
        if OWNED:save()
        raise
