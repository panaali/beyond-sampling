"""Capped certificate selection only; logs each cached/new bound once."""
from pathlib import Path
from collections import Counter
import hashlib,json,resource,sys,time
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from bound import Majorant,protocol as base_protocol
from selector import select

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    resource.setrlimit(resource.RLIMIT_CPU,(19,20));start=time.process_time()
    dest=P/'results.json';ledger=P/'bounds.jsonl';assert not dest.exists() and not ledger.exists()
    out=dict(complete=False,source_sha256=sha(__file__),selector_sha256=sha(P/'selector.py'),protocol_sha256=sha(P/'protocol.json'),groups=[],evaluations=0,cached=0,new=0)
    all_counts=Counter();new_counts=Counter()
    def save():
        out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            fresh_production_bound_counts=dict(all_counts),new_research_bound_counts=dict(new_counts))
        dest.write_text(json.dumps(out,indent=2)+'\n')
    save();ledger.touch(exist_ok=False)
    try:
        p=json.loads((P/'protocol.json').read_text())
        for name,h in p['pins'].items():assert sha(name)==h,name
        assert sha(P/'controls.json')=='e0ccad0c86f8d3dd7fc9a0bb0b907cc1e2319818341f79abefd65d60e699e1ad'
        controls=json.loads((P/'controls.json').read_text())
        assert controls['complete'] and controls['selector_sha256']==out['selector_sha256'] and controls['protocol_sha256']==out['protocol_sha256']
        assert controls['source_sha256']==sha(P/'controls.py')
        out['controls_sha256']=sha(P/'controls.json')
        bp=base_protocol();prior=json.loads((P.parent/'results.json').read_text());assert prior['complete']
        M=Majorant(bp);out['cold_counts']=M.cold_counts
        cached={};certs={};parents={}
        for row in prior['rows']:
            assert row['status']=='completed'
            key=(row['panel'],row['fixture'],row['side']);full=key+(row['order'],)
            assert full not in cached;cached[full]=row
            if key in certs:assert certs[key]==row['certificate']
            certs[key]=row['certificate'];parents.setdefault(key[:2],set()).add(key)
        assert len(cached)==99 and len(certs)==33 and len(parents)==22
        out['required_prior_certificate_advice']=prior['reused_certificate_setup']
        out['required_partition_advice']=str(P.parent.parent/'edge_split5/geometry.json')
        seen=set()
        def get(key,n):
            full=key+(n,);assert full not in seen;seen.add(full)
            rec=dict(index=out['evaluations'],key=list(key),order=n,certificate=certs[key],status='started')
            out['active_bound']=dict(rec);save()
            if full in cached:
                old=cached[full];rec.update(status='completed',origin='cached',upper_bound=old['upper_bound'],counts=old['counts'],prior_record=full)
                out['cached']+=1
            else:
                rec['origin']='new';before=Counter(M.D.counts);t=time.process_time()
                try:rec.update(status='completed',**M.one(certs[key],n))
                except Exception as e:rec.update(status='unknown',exception=repr(e),counts=dict(Counter(M.D.counts)-before))
                rec['research_cpu']=time.process_time()-t;out['new']+=1;new_counts.update(rec['counts'])
            all_counts.update(rec['counts']);out['evaluations']+=1
            with ledger.open('a') as f:f.write(json.dumps(rec)+'\n')
            out['active_bound']=None;save()
            return rec
        for panel,name in parents:
            keys=sorted(parents[(panel,name)],key=lambda x:-1 if x[2] is None else x[2]);assert len(keys)==(1 if panel=='original' else 2)
            row=dict(panel=panel,fixture=name,status='started');out['groups'].append(row);save()
            def emit(result):row['selection']=result;save()
            row['selection']=select(keys,p['orders'],p['tolerance'],get,emit);row['status']='completed';save()
        out['bounds_sha256']=sha(ledger)
        out['scope']='Selected orders are ideal-bound predictions only, not measured integration results. No quadrature or model calls. Kernel proxy, all bound counts and required prior geometry/range advice remain separate; native/cold/floating errors unresolved.'
        out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:save();print(json.dumps({k:out[k] for k in ('complete','evaluations','cached','new','cpu','max_rss_kib')}))
if __name__=='__main__':main()
