"""Independent saved-only contraction; never import or execute author sources."""
import hashlib,json,math,pathlib,resource,signal,time,traceback
from fractions import Fraction as F
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parent
START=time.process_time()
OUT=HERE/'results.json'
PROGRESS=HERE/'progress.jsonl'
STATE={'status':'started','records':[],'scope':'Saved arithmetic only; inherited CIs not reconstructed; aggregate scramble consistency is not per-scramble evidence.'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save():
    tmp=OUT.with_suffix('.tmp');tmp.write_text(json.dumps(STATE,indent=2,allow_nan=False)+'\n');tmp.replace(OUT)
def record(kind,value):
    STATE['records'].append({'kind':kind,**value});save()
    with PROGRESS.open('a') as f:f.write(json.dumps({'kind':kind,'index':len(STATE['records'])})+'\n')
def exact(q):return [q.numerator,q.denominator]
def near(x,q):
    assert type(x) in (int,float) and math.isfinite(x)
    z=F(x);assert abs(z-q)<=F(1,10**12)+F(1,10**10)*max(abs(z),abs(q)),(x,float(q))
def group(m):return 'orig' if m=='orig' else 'seed' if m.startswith('seed') else m.rsplit('_',1)[0]
def alarm(*_):raise TimeoutError('19 CPU-second independent audit limit')
def main():
    p=json.loads((HERE/'protocol.json').read_text());g=json.loads((HERE/'gate.json').read_text())
    assert g.get('status')=='passed', 'Independent gate not bound'
    for item in g['pins']:
        path=pathlib.Path(item['path']);assert sha(path)==item['sha256']
    assert sha(HERE/'check.py')==p['checker_sha256'];assert sha(ROOT/'manifest.json')==p['manifest_sha256']
    manifest=json.loads((ROOT/'manifest.json').read_text());assert manifest['author_commit']==p['author_commit']
    for row in manifest['files']:
        path=ROOT/row['snapshot_path'];b=path.read_bytes()
        assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['git_blob']
    base=ROOT/'frozen/m16/approaches/tie_blind'
    raw=[json.loads(s) for s in (base/'lb/raw.jsonl').read_text().splitlines()]
    tables=json.loads((base/'lb/table.json').read_text());pooled=json.loads((base/'lb/table_pooled.json').read_text());seeds=json.loads((base/'seeds.json').read_text())
    truth=json.loads((ROOT/'frozen/results/leaderboard_truths.json').read_text())
    names=['sci_tao','tie_blind'];budgets=[1000,4000,16000,64000,256000]
    models=['orig']+[f'seed{i}' for i in range(5)]+[f's{sg}_{i}' for sg in ['0.0025','0.005','0.01','0.02'] for i in range(6)]
    assert set(truth)==set(models) and len(raw)==300 and len(tables)==60 and len(pooled)==10 and len(seeds)==12
    expected={(a,m,b,0) for a in names for m in models for b in budgets}
    by={}
    for r in raw:
        key=(r['name'],r['model'],r['budget'],r['rep']);assert key in expected and key not in by
        assert (r['v'],r['H'],r['T'])==(3,16,10) and r['det_ok'] is True
        assert all(math.isfinite(r[k]) for k in ['est','cost','cpu_s']) and r['cost']>0 and r['cpu_s']>=0
        t=truth[r['model']];assert 0<t['acc']<1 and t['se']>=0 and type(t['n']) is int and t['n']>0
        # Independent rational arithmetic on stored binary64 values/constant.
        cost=F(r['cost'])*(F(2.23) if r['name']=='sci_tao' else 1)
        delta=F(r['est'])-F(t['acc']);e2=delta**2-F(t['se'])**2;v=F(t['acc'])*(1-F(t['acc']))/cost
        by[key]=(cost,delta,e2,v)
    assert set(by)==expected
    record('coverage',{'raw_rows':len(raw),'unique_keys':len(by),'models':models,'reference_n':{m:truth[m]['n'] for m in models},'snapshot_files':len(manifest['files'])})
    seen=set()
    for table_kind,arr in [('grouped',tables),('pooled',pooled)]:
        wanted={(a,b,g) for a in names for b in budgets for g in (['orig','seed','s0.0025','s0.005','s0.01','s0.02'] if table_kind=='grouped' else ['all30'])}
        keys=set()
        for r in arr:
            key=(r['name'],r['budget'],r['group']);assert key in wanted and key not in keys;keys.add(key)
            selected=[by[(r['name'],m,r['budget'],0)] for m in models if r['group']=='all30' or group(m)==r['group']]
            n=len(selected);assert r['n']==n and r['cls']=='quadrature' and r['tuned'] is False
            c,d,e,v=[sum((row[j] for row in selected),F(0))/n for j in range(4)]
            ratio=e/v;near(r['cost'],c);near(r['ratio'],ratio);near(r['bias_pp'],100*d);near(r['rmse_pp']**2,10000*max(e,F(0)))
            assert len(r['ci'])==2 and all(math.isfinite(x) for x in r['ci']) and r['ci'][0]<=r['ci'][1]
            assert math.isfinite(r['time_ratio']) and r['time_ratio']>=0
            if table_kind=='grouped':
                near(r['ratio_T196'],ratio*F(6000,5948));assert len(r['ci_T196'])==2
                for old,new in zip(r['ci'],r['ci_T196'],strict=True):near(new,F(old)*F(6000,5948))
                if r['group']=='orig':
                    for x in r['ci']:near(x,ratio)
            record(table_kind,{'key':key,'n':n,'ratio_exact':exact(ratio),'ratio':float(ratio),'cost_exact':exact(c),'corrected_mse_exact':exact(e),'original':r,'ci_status':'inherited; endpoint conversion checked only'})
        assert keys==wanted
    snames=['orig','seed0','s0.0025_0','s0.005_0','s0.01_0','s0.02_0'];assert set(seeds)=={f'{m}@{b}' for m in snames for b in [16000,64000]}
    for key,methods in seeds.items():
        m,b=key.split('@');assert set(methods)==set(names);t=truth[m];variance=F(t['acc'])*(1-F(t['acc']))
        for name,r in methods.items():
            assert set(r)=={'ratio','cost','rmse_pp'} and r['cost']>0 and all(math.isfinite(v) for v in r.values())
            mse_from_ratio=F(r['ratio'])*variance/F(r['cost'])
            near(r['rmse_pp']**2,10000*max(mse_from_ratio,F(0)))
            record('scramble_aggregate',{'key':[key,name],'original':r,'implied_corrected_mse_exact':exact(mse_from_ratio),'scope':'Internal aggregate consistency only; individual 16 values/costs and CI absent.'})
    assert len(STATE['records'])==95
    STATE['status']='passed';STATE['counts']={'raw':300,'grouped':60,'pooled':10,'scramble_aggregates':24}
    STATE['exclusions']=['No bootstrap or RNG','No model/estimator/checkpoint/library author calls','No reconstruction of per-scramble values','No original-time identity/full-cost/rung verification','No validation of inherited CI uncertainty or time ratios']
if __name__=='__main__':
    assert not OUT.exists() and not PROGRESS.exists();save()
    signal.signal(signal.SIGPROF,alarm);signal.setitimer(signal.ITIMER_PROF,19)
    try:main()
    except BaseException as e:STATE.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        signal.setitimer(signal.ITIMER_PROF,0);STATE['cpu_seconds']=time.process_time()-START;STATE['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;save()
