"""Sole exact synthetic-table controls; no model, arrays, author or RNG imports."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,os,resource,sys,time,traceback
from primitive import score,tail,smaller_half,permute_labels,validate
HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pack(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {json.dumps(k) if isinstance(k,tuple) else k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(y) for y in x]
    return x

def main():
    assert not OUT.exists(),'preserve all prior attempts'
    assert 0<resource.getrlimit(resource.RLIMIT_CPU)[0]<=20
    assert 0<resource.getrlimit(resource.RLIMIT_AS)[0]<=2147483648
    OUT.mkdir();start=time.process_time();r=dict(complete=False,cases=[],guards=[],model_face_rng_calls=0)
    def save():
        r.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        p=OUT/'results.tmp';p.write_text(json.dumps(pack(r),indent=2,allow_nan=False)+'\n')
        with p.open('rb') as f:os.fsync(f.fileno())
        os.replace(p,OUT/'results.json')
    def event(phase,**kw):
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,**kw))+'\n');f.flush();os.fsync(f.fileno())
    def cap():assert time.process_time()-start<17,'17CPU cap; no automatic retry'
    save();event('attempt',name='preflight')
    try:
        p=read(HERE/'protocol.json');g=read(HERE/'gate.json')
        assert g['clearance'] is True and g['protocol_sha256']==sha(HERE/'protocol.json')
        for pins in (p['pins'],g['pins']):
            for path,digest in pins.items():assert sha(path)==digest,path
        required=[str(HERE/x) for x in ('primitive.py','controls.py','fixtures.json','protocol.json')]+[p['design_review'],g['source_review']['path']]
        assert all(x in g['pins'] for x in required)
        assert len(g['source_review']['publication'])>=8 and all(x in '0123456789abcdef' for x in g['source_review']['publication'])
        design=read(p['design_review']);assert design['complete'] and design['pass_all'] and design['execution_authorized'] is False
        sr=read(g['source_review']['path']);assert sr['complete'] and sr['pass_all']
        assert all(sr['pins'][str(HERE/x)]==sha(HERE/x) for x in ('primitive.py','controls.py','fixtures.json','protocol.json'))
        r.update(runtime=dict(executable=sys.executable,prefix=sys.prefix),gate=g,protocol_sha256=sha(HERE/'protocol.json'));save()
        assert r['runtime']==p['runtime']
        fixtures=read(HERE/'fixtures.json');assert len(fixtures['cases'])==8 and len(fixtures['guards'])==10
        event('result',name='preflight')
        def decode(rows):return [dict(x,value=F(x['value'])) for x in rows]
        for c in fixtures['cases']:
            cap();event('attempt',name=c['name']);row=dict(name=c['name']);r['cases'].append(row);save()
            A=decode(c['A']);B=decode(c['B']);ans=score(A,B);row['score']=ans;save()
            exp=c['expected']
            for key in ('mse','baseline_mse','ratio'):
                assert ans[key]==(None if exp[key] is None else F(exp[key])),(c['name'],key)
            assert pack(ans['unseen'])==exp.get('unseen',[])
            if 'tail' in exp:
                ids=tail(B,F(exp['tail_fraction']));row['tail']=ids;save();assert pack(ids)==exp['tail']
            if 'permutation' in exp:
                perm=permute_labels(B,exp['permutation']);row['permuted']=perm;save()
                original=Counter((x['kind'],x['label']) for x in B);changed=Counter((x['kind'],x['label']) for x in perm)
                row['occupancy_preserved']=original==changed;save();assert original==changed
                assert all(a['value']==b['value'] and a['index']==b['index'] for a,b in zip(B,perm,strict=True))
            if 'smaller_tail' in exp:
                small=smaller_half(B);ids=tail(small,F(exp['tail_fraction']));smallids={(x['kind'],x['index']) for x in small}
                intersection=[x for x in row['tail'] if x in smallids]
                row.update(smaller=small,smaller_tail=ids,old_tail_intersection=intersection);save()
                assert pack(ids)==exp['smaller_tail'] and pack(intersection)==exp['old_tail_intersection']
            if c['name']=='independent_95pct':
                total=sum((x['value']**2 for x in B),F(0));covered=sum((x['value']**2 for x in B if x['label']=='common'),F(0))
                row['positive_square_coverage']=covered/total;save();assert covered/total==F(19,20)
            row['passed']=True;save();event('result',name=c['name'],passed=True)
        base=decode(fixtures['cases'][0]['A'])
        for name in fixtures['guards']:
            cap();event('attempt',name=name);rows=[dict(x) for x in base]
            if name=='empty':rows=[]
            if name=='negative':rows[0]['value']=F(-1)
            if name=='float_nan':rows[0]['value']=float('nan')
            if name=='duplicate_identity':rows[1]['index']=rows[0]['index']
            if name=='unknown_status':rows[0]['status']='capped'
            if name=='missing_label':rows[0]['label']=None
            if name=='permutation_crosskind':rows[1]['kind']='23'
            if name=='half_odd':rows=rows[:1]
            caught=None;returned=None
            try:
                if name=='permutation_crosskind':returned=permute_labels(rows,[1,0])
                elif name=='permutation_duplicate':returned=permute_labels(rows,[0,0])
                elif name=='tail_zero':returned=tail(rows,F(0))
                elif name=='half_odd':returned=smaller_half(rows)
                else:returned=validate(rows)
            except Exception as e:caught=dict(type=type(e).__name__,message=str(e))
            row=dict(name=name,caught=caught,unexpected_return_repr=None if caught else repr(returned));r['guards'].append(row);save()
            assert caught is not None and caught['type']=='ValueError'
            row['passed']=True;save();event('result',name=name,passed=True)
        assert len(r['cases'])==8 and len(r['guards'])==10
        r['complete']=True;r['pass_all']=True;event('complete');save()
    except BaseException as e:
        r.update(failure=repr(e),traceback=traceback.format_exc());event('failure',error=repr(e));save();raise
if __name__=='__main__':main()
