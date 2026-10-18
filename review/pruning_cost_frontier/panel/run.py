"""Fixed synthetic controls, no retained target data or scientific libraries."""
import hashlib,json,resource,time,traceback
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent

def enc(v):
    if isinstance(v,F):return [str(v.numerator),str(v.denominator)]
    if isinstance(v,dict):return {k:enc(x) for k,x in v.items()}
    if isinstance(v,list):return [enc(x) for x in v]
    return v

def main():
    out=P/'results.json';prog=P/'progress.jsonl';assert not out.exists() and not prog.exists()
    start=time.process_time();r={'complete':False,'pass_all':False,'rows':[],'guards':[],'calls_attempted':0,'calls_returned':0}
    def save():
        r['cpu_seconds']=time.process_time()-start;tmp=P/'results.tmp';tmp.write_text(json.dumps(enc(r),indent=2)+'\n');tmp.replace(out)
    def event(kind,name):
        with prog.open('a') as f:f.write(json.dumps({'event':kind,'name':name,'cpu_seconds':time.process_time()-start})+'\n')
        if time.process_time()-start>=17:raise RuntimeError('internal CPU cap')
    try:
        p=json.loads((P/'protocol.json').read_text());g=json.loads((P/'gate.json').read_text())
        required=[str(P/n) for n in ('primitive.py','run.py','fixtures.json','protocol.json','PROTOCOL.md')]
        assert g['ready'] is True and all(f in g['pins'] for f in required)
        assert g['source_review'] in g['pins']
        for key in ['source_review_commit','source_review_publication','source_publication']:
            assert isinstance(g[key],str) and 8<=len(g[key])<=40 and all(c in '0123456789abcdef' for c in g[key])
        for pins in [p['pins'],g['pins']]:
            for f,h in pins.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
        rv=json.loads(Path(g['source_review']).read_text());assert all(rv[k] is True for k in ['complete','pass_all','execution_allowed'])
        assert all(f in rv['pins'] for f in required)
        for f,h in rv['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
        proof=json.loads(Path(p['proof_review']).read_text());assert proof['complete'] is True and proof['pass_all'] is True
        assert p['proof_review'] in p['pins']
        r['limits']={'cpu':resource.getrlimit(resource.RLIMIT_CPU),'address_space':resource.getrlimit(resource.RLIMIT_AS)};save()
        assert r['limits']=={'cpu':(20,20),'address_space':(2**31,2**31)}
        f=json.loads((P/'fixtures.json').read_text());assert len(f['valid'])==12 and len(f['guards'])==6 and f['bits']==80
        prior=json.loads(Path(p['prior_results']).read_text());audit=json.loads(Path(p['prior_saved_review']).read_text())
        assert prior['complete'] is True and prior['pass_all'] is True and audit['complete'] is True and audit['pass_all'] is True
        assert p['prior_results'] in p['pins'] and p['prior_saved_review'] in p['pins']
        assert audit['pins'][p['prior_results']]==p['pins'][p['prior_results']]
        assert len(prior['rows'])==12 and len(prior['guards'])==6
        from primitive import frontier
        event('start','controls')
        for case,oldrow in zip(f['valid'],prior['rows'],strict=True):
            event('attempt',case['name']);r['calls_attempted']+=1;save()
            ans=frontier(case['e'],case['c'],case['O'],case['Q'],f['bits']);r['calls_returned']+=1
            row={'name':case['name'],'status':'returned_unchecked','raw':ans};r['rows'].append(row);save()
            assert oldrow['name']==case['name'] and oldrow['status']=='passed'
            assert enc(ans)==oldrow['raw']
            row['all_raw_fields_equal']=True;save()
            opt=F(case['optimum']);assert ans['fractional_saved_lower']<=opt<=ans['fractional_saved_upper']
            assert ans['fractional_saved_upper']-ans['fractional_saved_lower']<=ans['removable']/2**80
            if 'expected_method_saved' in case:
                for n,v in case['expected_method_saved'].items():assert ans['methods'][n]['best']['saved']==F(v)
            if 'probe' in case:
                v=case['probe'];row['probe_feasible']=F(v['error'])**2*(F(case['O'])-F(v['saved']))<=F(case['Q']);save();assert row['probe_feasible'] is v['feasible']
            row['status']='passed';save();event('result',case['name'])
        for case,oldrow in zip(f['guards'],prior['guards'],strict=True):
            event('attempt',case['name']);r['calls_attempted']+=1;save()
            try:ans=frontier(case['e'],case['c'],case['O'],case['Q'],f['bits'])
            except ValueError as exc:r['guards'].append({'name':case['name'],'rejected':True,'message':str(exc)});save()
            else:
                r['calls_returned']+=1;r['guards'].append({'name':case['name'],'rejected':False,'raw':ans});save();raise AssertionError('guard accepted')
            assert r['guards'][-1]==oldrow
            event('result',case['name'])
        assert r['calls_attempted']==18 and r['calls_returned']==12
        event('complete','controls');r.update(complete=True,pass_all=True);save()
    except BaseException:
        r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
