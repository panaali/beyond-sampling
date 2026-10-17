"""Bound-only replay of33 saved certificates at three frozen orders."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,time,resource
from bound import protocol,Majorant,sha
P=Path(__file__).resolve().parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22))
    dest=P/'results.json';assert not dest.exists();start=time.process_time()
    out=dict(complete=False,source_sha256=sha(__file__),bound_sha256=sha(P/'bound.py'),protocol_sha256=sha(P/'protocol.json'),rows=[],groups=[])
    def save():
        out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dest.write_text(json.dumps(out,indent=2)+'\n')
    save()
    try:
        p=protocol();controls=json.loads((P/'controls.json').read_text())
        assert controls['complete'] and controls['protocol_sha256']==out['protocol_sha256'] and controls['bound_sha256']==out['bound_sha256']
        assert controls['source_sha256']==sha(P/'controls.py')
        out['controls_sha256']=sha(P/'controls.json')
        original=json.loads(Path(p['original']).read_text());children=json.loads(Path(p['children']).read_text())
        assert original['complete'] and children['complete']
        names=[c['fixture'] for c in original['certificates']];assert len(names)==len(set(names))==11
        assert len(children['children'])==22 and {(c['fixture'],c['side']) for c in children['children']}=={(n,s) for n in names for s in (0,1)}
        certs=[dict(panel='original',fixture=c['fixture'],side=None,certificate=c) for c in original['certificates']]
        certs.extend(dict(panel='child',fixture=c['fixture'],side=c['side'],certificate=c['certificate']) for c in children['children'])
        out['reused_certificate_count']=len(certs)
        out['reused_certificate_setup']=[dict(panel=c['panel'],fixture=c['fixture'],side=c['side'],counts=c['certificate']['counts'],max_bits=c['certificate']['max_bits'],prior_cpu=c['certificate']['cpu']) for c in certs]
        M=Majorant(p);out['cold_counts']=M.cold_counts
        for item in certs:
            for order in p['orders']:
                row=dict(panel=item['panel'],fixture=item['fixture'],side=item['side'],order=order,status='started',certificate={k:item['certificate'][k] for k in ('a','b','det')})
                out['rows'].append(row);save()
                try:
                    result=M.one(item['certificate'],order)
                    if item['panel']=='original':
                        old=next(r for r in original['rows'] if r['fixture']==item['fixture'] and r['order']==order)
                        old_bound=old['upper_bound'];saved_error=str(old['source_absolute_error'])
                    else:
                        old=next(r for r in children['rows'] if r['fixture']==item['fixture'] and r['side']==item['side'] and r['order']==order)
                        old_bound=old['ideal_rule_bound']['upper_bound'];saved_error=old['absolute_error']
                    row.update(status='completed',**result,previous_bound=old_bound,strictly_tighter=F(result['upper_bound'])<F(old_bound),
                        saved_floating_absolute_error=saved_error,saved_error_below_ideal_bound=F(saved_error)<=F(result['upper_bound']))
                except Exception as e:row.update(status='unresolved',exception=repr(e))
                save()
        assert len(out['rows'])==99
        for panel in ('original','child'):
            for name in names:
                for order in p['orders']:
                    rows=[r for r in out['rows'] if r['panel']==panel and r['fixture']==name and r['order']==order]
                    gr=dict(panel=panel,fixture=name,order=order,complete=len(rows)==(1 if panel=='original' else 2) and all(r['status']=='completed' for r in rows))
                    if gr['complete']:
                        new=sum(F(r['upper_bound']) for r in rows);old=sum(F(r['previous_bound']) for r in rows)
                        gr.update(bound_sum_fraction=str(new),previous_sum_fraction=str(old),qualifies=new<=F(p['tolerance']),strictly_tighter=new<old,
                            new_over_previous=str(new/old) if old else None)
                    out['groups'].append(gr)
        out['scope']='All bounds are ideal-rule truncation only. Reused certificate advice is not free production computation; no mass/geometry/model rerun. Saved floating error comparisons are diagnostics, not roundoff certificates.'
        out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:save();print(json.dumps({k:out[k] for k in ('complete','cpu','max_rss_kib')}))
if __name__=='__main__':main()
