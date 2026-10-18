"""Saved JSON parity only; neither optimizer is imported or invoked."""
import hashlib,json,resource,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def verify(pins):
    for p,h in pins.items():assert sha(p)==h,p
def main():
    out=P/'results.json';progress=P/'progress.jsonl'
    assert not out.exists() and not progress.exists()
    start=time.process_time();result={'complete':False,'pass_all':False,'valid':[],'guards':[]}
    def save():
        result['cpu_seconds']=time.process_time()-start
        tmp=P/'results.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out)
    def event(name):
        with progress.open('a') as f:f.write(json.dumps({'event':name,'cpu_seconds':time.process_time()-start})+'\n')
        assert time.process_time()-start<17
    try:
        p=read(P/'protocol.json');g=read(P/'gate.json');verify(p['pins']);verify(g['pins'])
        assert g['ready'] is True and g['review'] in g['pins']
        for k in ['review_commit','review_publication','source_publication']:
            v=g[k];assert isinstance(v,str) and 8<=len(v)<=40 and all(c in '0123456789abcdef' for c in v)
        review=read(g['review']);assert all(review[k] is True for k in ['complete','pass_all','execution_allowed']);verify(review['pins'])
        for n in ['check.py','protocol.json','PROTOCOL.md']:assert str(P/n) in g['pins'] and str(P/n) in review['pins']
        B=Path(p['producer']);protocol=read(B/'protocol.json');pg=read(B/'gate.json')
        verify(protocol['pins']);verify(pg['pins']);assert pg['ready'] is True
        sr=read(pg['source_review']);verify(sr['pins']);assert all(sr[k] is True for k in ['complete','pass_all','execution_allowed'])
        result['limits']={'cpu':list(resource.getrlimit(resource.RLIMIT_CPU)),'address_space':list(resource.getrlimit(resource.RLIMIT_AS))};save()
        assert result['limits']=={'cpu':[20,20],'address_space':[2**31,2**31]}
        old=read(protocol['prior_results']);new=read(B/'results.json');audit=read(protocol['prior_saved_review']);f=read(B/'fixtures.json')
        assert audit['complete'] is True and audit['pass_all'] is True and audit['pins'][protocol['prior_results']]==sha(protocol['prior_results'])
        for r in [old,new]:
            assert r['complete'] is True and r['pass_all'] is True and 'failure' not in r
            assert len(r['rows'])==12 and len(r['guards'])==6 and r['calls_attempted']==18 and r['calls_returned']==12
            assert r['limits']==result['limits'] and 0<=r['cpu_seconds']<17
        assert len(f['valid'])==12 and len(f['guards'])==6 and f['bits']==80
        event('begin')
        for a,b,c in zip(old['rows'],new['rows'],f['valid'],strict=True):
            assert a['name']==b['name']==c['name'] and a['status']==b['status']=='passed'
            assert b['all_raw_fields_equal'] is True
            comparable={k:v for k,v in b.items() if k!='all_raw_fields_equal'}
            assert canonical(a)==canonical(comparable)
            digest=hashlib.sha256(canonical(a['raw']).encode()).hexdigest()
            result['valid'].append({'name':c['name'],'all_fields_equal':True,'canonical_raw_sha256':digest});save();event(c['name'])
        for a,b,c in zip(old['guards'],new['guards'],f['guards'],strict=True):
            assert a['name']==b['name']==c['name'] and a['rejected'] is True and b['rejected'] is True
            assert canonical(a)==canonical(b)
            result['guards'].append({'name':c['name'],'all_fields_equal':True});save();event(c['name'])
        events=[json.loads(x) for x in (B/'progress.jsonl').read_text().splitlines()]
        expected=[('start','controls')]+[(kind,c['name']) for c in f['valid']+f['guards'] for kind in ['attempt','result']]+[('complete','controls')]
        assert len(events)==38 and [(e['event'],e['name']) for e in events]==expected
        times=[e['cpu_seconds'] for e in events];assert all(0<=a<=b for a,b in zip(times,times[1:])) and times[-1]<=new['cpu_seconds']
        result.update(attempts=18,returns=12,producer_events=38,producer_cpu_seconds=new['cpu_seconds'],prior_audit=str(protocol['prior_saved_review']),pins=p['pins'],helper_calls=0,target_payload_reads=0)
        event('complete');assert len(progress.read_text().splitlines())==20
        result.update(complete=True,pass_all=True);save()
    except BaseException:
        result.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
