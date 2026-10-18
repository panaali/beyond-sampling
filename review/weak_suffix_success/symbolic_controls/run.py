"""Frozen synthetic control harness: no CDF or trained model calls."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,os,resource,time,traceback
HERE=Path(__file__).resolve().parent; OUT=HERE/'output'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def serial(x):
    if isinstance(x,F): return {'q':[x.numerator,x.denominator]}
    if isinstance(x,dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return list(map(serial,x))
    return x

def verify(pins):
    assert isinstance(pins,dict) and pins
    for p,h in pins.items():
        assert type(h)is str and len(h)==64 and all(c in '0123456789abcdef' for c in h)
        assert sha(p)==h,p

def hexid(v): return type(v)is str and len(v)>=8 and all(c in '0123456789abcdef' for c in v)

def direct(m,h,y):
    nxt=[]
    for k in range(m['H']):
        z=F(m['Wi'][k])*y
        for j in range(m['H']): z += F(m['Wh'][k][j])*h[j]
        nxt.append(max(F(0),z))
    logits=[]
    for row in m['Wo']:
        z=F(0)
        for k in range(m['H']): z+=F(row[k])*nxt[k]
        logits.append(z)
    best=0
    for j in range(1,len(logits)):
        if logits[j]>logits[best]:best=j
    return nxt,logits,best

def truth(m,y):
    a,b,ia,ib=m['rank']; rows=[(F(a),ia),(F(b),ib),(y,m['T']-1)]
    rows.sort(key=lambda z:(-z[0],z[1]));return rows[1][1]

def main():
    assert not OUT.exists() or not any(OUT.iterdir()),'refuse any existing output'
    OUT.mkdir(exist_ok=True);start=time.process_time()
    r=dict(complete=False,pass_all=False,rows=[],events=0,partition_attempts=0,partition_returns=0,guard_attempts=0,point_forwards=0,prefix_steps=0,cdf_calls=0,trained_calls=0,rng_calls=0)
    def save():
        r['cpu_seconds']=time.process_time()-start;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp=OUT/'results.tmp';tmp.write_text(json.dumps(serial(r),indent=2,allow_nan=False)+'\n');os.replace(tmp,OUT/'results.json')
    def cap():assert time.process_time()-start<15,'15 CPU internal cap'
    def event(kind,**kw):
        cap()
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,**kw))+'\n')
        r['events']+=1;save()
    try:
        save();p=read(HERE/'protocol.json');g=read(HERE/'gate.json')
        directpaths=[str(HERE/n) for n in ('primitive.py','fixtures.py','run.py','launch.sh','PROTOCOL.md','README.md','protocol.json')]
        assert g['ready'] is True and all(x in g['pins'] for x in directpaths+[p['written_review'],p['source_review']])
        assert all(hexid(g[k]) for k in ('source_review_commit','source_review_publication','execution_scope_publication'))
        verify(p['pins']);verify(g['pins']);sr=read(p['source_review']);wr=read(p['written_review'])
        assert sr['complete'] is True and sr['pass_all'] is True and sr['execution_authorized'] is True
        assert all(x in sr['pins'] for x in directpaths);verify(sr['pins'])
        assert wr['complete'] is True and wr['pass_all'] is True and wr['execution_allowed'] is False;verify(wr['pins'])
        assert resource.getrlimit(resource.RLIMIT_CPU)==(16,20) and resource.getrlimit(resource.RLIMIT_AS)==(2*1024**3,)*2
        r.update(pins=g['pins'],protocol_pins=p['pins'],limits=dict(cpu=[16,20],address_space=[2*1024**3]*2));event('started')
        from fixtures import cases,guards
        from primitive import partition
        cs=cases();gs=guards();assert len(cs)==10 and len(gs)==6
        for case in cs:
            row=dict(label=case['label'],fixture=case,status='running',references=[],prefix_references=[]);r['rows'].append(row);event('started_case',label=case['label'])
            r['partition_attempts']+=1;save();ans=partition(case['model']);row['answer']=ans;r['partition_returns']+=1;save()
            assert ans['symbolic']==case['expected']
            assert len(ans['cells'])==len(ans['roots'])+1 and ans['cells'][0]['l'] is None and ans['cells'][-1]['u'] is None
            for left,right in zip(ans['cells'],ans['cells'][1:]):assert left['u']==right['l']
            for cell in ans['cells']:
                cap();r['point_forwards']+=1;save();nxt,logits,pred=direct(case['model'],case['model']['h'],cell['y']);tr=truth(case['model'],cell['y'])
                row['references'].append(dict(kind='interval',y=cell['y'],hidden=nxt,logits=logits,prediction=pred,truth=tr));save()
                assert logits==cell['logits'] and pred==cell['prediction'] and tr==cell['truth']
            for y in ans['roots']:
                cap();r['point_forwards']+=1;save();nxt,logits,pred=direct(case['model'],case['model']['h'],y);tr=truth(case['model'],y)
                row['references'].append(dict(kind='boundary',y=y,hidden=nxt,logits=logits,prediction=pred,truth=tr));save()
                # Boundary predictions match both continuous affine extensions;
                # rank equality uses deterministic first-index sorting, zero mass.
                adjacent=[z for z in ans['gate_pieces'] if (z['l'] is None or z['l']<=y) and (z['u'] is None or y<=z['u'])]
                for z in adjacent:
                    vals=[a+b*y for a,b in zip(z['intercept'],z['slope'])]
                    assert vals==logits and max(range(len(vals)),key=vals.__getitem__)==pred
            if case['prefix'] is not None:
                h=[F(0)]*case['model']['H']
                for y in case['prefix']:
                    r['prefix_steps']+=1;save();h,logits,pred=direct(case['model'],h,F(y));row['prefix_references'].append(dict(y=y,hidden=h,logits=logits,prediction=pred));save()
                assert h==case['model']['h']
                order=sorted(range(len(case['prefix'])),key=lambda i:(-case['prefix'][i],i));ia,ib=order[:2]
                assert case['model']['rank']==[case['prefix'][ia],case['prefix'][ib],ia,ib]
            row['status']='passed';event('completed_case',label=case['label'])
        assert r['rows'][6]['answer']['symbolic']==r['rows'][7]['answer']['symbolic']
        assert r['rows'][8]['answer']['symbolic']!=r['rows'][9]['answer']['symbolic']
        for case in gs:
            row=dict(label=case['label'],fixture=case,status='running');r['rows'].append(row);event('started_guard',label=case['label']);r['guard_attempts']+=1;save()
            try:partition(case['model'])
            except ValueError as e:row['rejection']=str(e);save()
            else:raise AssertionError('guard accepted malformed input')
            row['status']='passed';event('completed_guard',label=case['label'])
        assert r['partition_attempts']==r['partition_returns']==10 and r['guard_attempts']==6 and r['prefix_steps']==3
        assert r['point_forwards']<=400 and len(r['rows'])==16
        event('completed');assert r['events']==34;r['complete']=r['pass_all']=True;save()
    except BaseException:
        r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
