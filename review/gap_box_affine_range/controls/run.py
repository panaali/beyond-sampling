"""One frozen synthetic attempt; imports candidate only after all published gates."""
from pathlib import Path
from fractions import Fraction as F
import hashlib, json, os, resource, time, traceback
HERE=Path(__file__).resolve().parent
OUT=HERE/'output'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return list(map(serial,x))
    return x

def main():
    assert not OUT.exists() or not any(OUT.iterdir()), 'refuse any existing output'
    OUT.mkdir(exist_ok=True);start=time.process_time()
    r=dict(complete=False,pass_all=False,rows=[],candidate_attempts=0,valid_returns=0,
           reference_attempts=0,reference_returns=0,guard_rejections=0,events=0,
           model_calls=0,cdf_calls=0,rng_calls=0)
    def save():
        r['cpu_seconds']=time.process_time()-start
        r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp=OUT/'results.tmp';tmp.write_text(json.dumps(serial(r),indent=2,allow_nan=False)+'\n')
        os.replace(tmp,OUT/'results.json')
    def budget():assert time.process_time()-start<15, 'internal CPU cap'
    def event(kind,**kw):
        budget()
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,**kw))+'\n')
        r['events']+=1;save()
    def checked_pins(pins):
        assert isinstance(pins,dict) and pins
        for p,h in pins.items():
            assert isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h)
            assert sha(p)==h,p
    def commit(s):assert isinstance(s,str) and len(s)>=8 and all(c in '0123456789abcdef' for c in s)
    try:
        save();protocol=json.loads((HERE/'protocol.json').read_text());gate=json.loads((HERE/'gate.json').read_text())
        direct=[HERE/x for x in ('primitive.py','fixtures.py','reference.py','run.py','launch.sh','PROTOCOL.md','protocol.json')]
        assert gate['ready'] is True
        for k in ('source_review_commit','source_review_publication','execution_scope_publication'):commit(gate[k])
        assert all(str(p) in gate['pins'] for p in direct+[Path(protocol['source_review']),Path(protocol['theorem_review'])])
        checked_pins(protocol['pins']);checked_pins(gate['pins'])
        review=json.loads(Path(protocol['source_review']).read_text())
        assert review['complete'] is True and review['pass_all'] is True and review['execution_authorized'] is True
        assert all(str(p) in review['pins'] for p in direct)
        checked_pins(review['pins'])
        theorem=json.loads(Path(protocol['theorem_review']).read_text())
        assert theorem['complete'] is True and theorem['pass_all'] is True
        cpu=resource.getrlimit(resource.RLIMIT_CPU);mem=resource.getrlimit(resource.RLIMIT_AS)
        assert all(0<x<=20 for x in cpu) and all(0<x<=2*1024**3 for x in mem)
        r['limits']=dict(cpu=cpu,address_space=mem);r['pins']=gate['pins'];r['protocol_pins']=protocol['pins']
        event('started')
        from primitive import affine_range,from_saved_v
        from reference import vertices
        from fixtures import CASES,GUARDS,BASE
        assert len(CASES)==12 and len(GUARDS)==8
        keys=('box','coeff','constant','leader','second','gap')
        def call(row,key,fn,args):
            budget();row['active_call']=key;r['candidate_attempts']+=1;save()
            value=fn(**args);row[key]=value;r['valid_returns']+=1;save();return value
        for case in CASES:
            row=dict(kind='domain',name=case['name'],input=case,status='started');r['rows'].append(row);save();event('started_case',name=case['name'])
            args={k:case[k] for k in keys};physical=call(row,'physical',affine_range,args)
            budget();row['active_call']='vertex_reference';r['reference_attempts']+=1;save()
            ref=vertices(**args);row['vertex_reference']=ref;r['reference_returns']+=1;save()
            assert physical['range']==ref['range']
            if 'expected' in case:assert physical['range']==(None if case['expected'] is None else list(map(F,case['expected'])))
            for candidate in physical['candidates']:
                for witness,value in zip((candidate['low'],candidate['high']),candidate['values'],strict=True):
                    i,j=case['leader'],case['second'];g=F(case['gap'])
                    assert all(F(l)<=x<=F(u) for (l,u),x in zip(case['box'],witness,strict=True))
                    assert witness[i]-witness[j]>=g and all(witness[j]-witness[k]>=g for k in range(len(witness)) if k not in(i,j))
                    assert F(case['constant'])+sum((F(a)*x for a,x in zip(case['coeff'],witness,strict=True)),F(0))==value
            vb=[[2*F(v) for v in pair] for pair in case['box']];va=[F(a)/2 for a in case['coeff']]
            vargs=dict(args,box=vb,coeff=va,gap=2*F(case['gap']))
            v=call(row,'v_domain',affine_range,vargs)
            assert v['range']==physical['range'] and v['status']==physical['status']
            assert [t/2 for t in v['second_interval']]==physical['second_interval']
            assert len(v['candidates'])==len(physical['candidates'])
            for vr,pr in zip(v['candidates'],physical['candidates'],strict=True):
                assert vr['t']/2==pr['t'] and vr['values']==pr['values']
                assert all([x/2 for x in vr[k]]==pr[k] for k in ('low','high'))
            adapted=call(row,'saved_v_adapter',from_saved_v,dict(box_v=vb,coeff_v=va,constant=case['constant'],leader=case['leader'],second=case['second'],gap_x=case['gap']))
            assert adapted==physical
            if 'extremum_t' in case:
                side,t=case['extremum_t'];index=0 if side=='min' else 1;t=F(t)
                assert physical['second_interval'][0]<t<physical['second_interval'][1]
                assert any(z['t']==t and z['values'][index]==physical['range'][index] for z in physical['candidates'])
            row['status']='passed';row['active_call']=None;save();event('completed_case',name=case['name'])
        rows={x['name']:x for x in r['rows']}
        assert rows['mixed_four']['physical']['range']==rows['permuted_four']['physical']['range']
        for name,change in GUARDS:
            args=dict(BASE,**change);row=dict(kind='guard',name=name,input=args,status='started',expected='ValueError');r['rows'].append(row);save();event('started_case',name=name)
            r['candidate_attempts']+=1;row['attempted']=True;save();budget()
            try:value=affine_range(**args)
            except ValueError as e:row['error']=str(e);r['guard_rejections']+=1
            else:row['unexpected_result']=value;save();raise AssertionError('guard accepted '+name)
            row['status']='passed';save();event('completed_case',name=name)
        assert len(r['rows'])==20 and all(x['status']=='passed' for x in r['rows'])
        assert (r['candidate_attempts'],r['valid_returns'],r['reference_attempts'],r['reference_returns'],r['guard_rejections'])==(44,36,12,12,8)
        event('completed');assert r['events']==42
        r['complete']=r['pass_all']=True;save()
    except BaseException:
        r['complete']=r['pass_all']=False;r['failure']=traceback.format_exc();save();raise

if __name__=='__main__':main()
