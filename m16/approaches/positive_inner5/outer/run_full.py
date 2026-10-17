"""Fixed99 rows, one saved pilot reused exactly once; no adaptive choices."""
from pathlib import Path
from decimal import Decimal as D,localcontext
from collections import Counter
import hashlib,json,time,resource
from common import load,rules,fixtures,one,HERE,json_safe

def annotate(row):
    if row['complete'] and row.get('finite_positive',False):
        with localcontext() as c:
            c.prec=150
            error=D.from_float(row['value'])-D(row['reference'])
            row.update(error=str(error),absolute_error=str(abs(error)),relative_error=str(abs(error/D(row['reference']))),
                       absolute_pass=abs(error)<=D('1e-10'),relative_pass=abs(error/D(row['reference']))<=D('1e-6'))
    return row

def main():
    dest=HERE/'full.json';progress=HERE/'full_progress.jsonl'
    assert not dest.exists() and not progress.exists()
    p=load();fp=json.loads((HERE/'full_protocol.json').read_text())
    assert not fp['pilot_review'].startswith('pending'), 'independent pilot review must be pinned before execution'
    for path,sha in fp['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    pilot=json.loads((HERE/'pilot.json').read_text());assert pilot['pass']
    assert pilot['row']['id']=='original/positive5orthant' and pilot['row']['order']==16
    assert pilot['row']['complete'] and pilot['row']['inner_calls']==16**3
    start=time.process_time();rs,setup=rules(p);manifest=fixtures(p)
    out={'complete':False,'full_protocol_sha256':hashlib.sha256((HERE/'full_protocol.json').read_bytes()).hexdigest(),
         'rows':[],'groups':[],'run_setup':setup,'pilot_setup':pilot['setup'],
         'canonical_mapping_named_arithmetic':96,'actually_executed_staged_mapping_named_arithmetic':192,
         'reused_pilot_rows':0,'scope':'fixed product diagnostics only; ideal inner functional distinct from actual floating implementation'}
    def record(x):
        with progress.open('a') as f:f.write(json.dumps(json_safe(x),allow_nan=False)+'\n')
    def save():
        out.update(cpu=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(json_safe(out),indent=2,allow_nan=False)+'\n');tmp.replace(dest)
    try:
        for n in p['orders']:
            for item in manifest:
                key=(item['id'],n)
                record({'stage':'attempt','id':item['id'],'order':n})
                if key==('original/positive5orthant',16):
                    row=json.loads(json.dumps(pilot['row']));row['reused_pilot']=True;out['reused_pilot_rows']+=1
                else:
                    row=one(item,n,rs[n]);row['reused_pilot']=False
                annotate(row);out['rows'].append(row);record({'stage':'row','row':row});save()
        assert len(out['rows'])==99 and out['reused_pilot_rows']==1
        assert len({(r['id'],r['order']) for r in out['rows']})==99
        original=json.loads(Path(p['original_source']).read_text());children=json.loads(Path(p['child_source']).read_text())
        parents={r['name']:r for r in original['fixtures']}
        for n in p['orders']:
            for kind in ['original','split']:
                for name,parent in parents.items():
                    selected=[r for r in out['rows'] if r['order']==n and r['kind']==kind and r['fixture']==name]
                    assert len(selected)==(1 if kind=='original' else 2)
                    good=all(r['complete'] and r.get('finite_positive',False) for r in selected)
                    counts=Counter()
                    for row in selected:
                        counts.update(row['inner_counts']);counts.update(row['construction_assembly_counts'])
                    group={'order':n,'kind':kind,'fixture':name,'complete':good,
                           'inner_calls':sum(r['inner_calls'] for r in selected),'named_counts':dict(counts),
                           'opaque_det5_calls':len(selected),'assembly_additions':len(selected)-1}
                    if good:
                        with localcontext() as c:
                            c.prec=150
                            value=sum(D.from_float(r['value']) for r in selected);error=value-D(parent['reference120'])
                            group.update(exact_float_sum=str(value),reference=parent['reference120'],error=str(error),
                                         absolute_error=str(abs(error)),relative_error=str(abs(error/D(parent['reference120']))),
                                         absolute_pass=abs(error)<=D('1e-10'),relative_pass=abs(error/D(parent['reference120']))<=D('1e-6'))
                    if kind=='original':
                        prior=next(r for r in original['rows'] if r['fixture']==name and r['arm']=='duffy'+str(n))
                        group['prior_4d_rule']={k:prior.get(k) for k in ['value','error','absolute_error','relative_error','source_proxy_ops']}
                    else:
                        prior=next(r for r in children['groups'] if r['fixture']==name and r['order']==n)
                        group['prior_4d_rule']={k:prior.get(k) for k in ['error','absolute_error','relative_error','source_proxy_ops','nodes']}
                    out['groups'].append(group)
        out['complete']=True
        out['all_numeric_rows_returned']=all(r['complete'] and r.get('finite_positive',False) for r in out['rows'])
    except BaseException as exc:
        out['failure']=repr(exc);record({'stage':'failure','exception':repr(exc)});raise
    finally:
        save()
    print(json.dumps({'complete':out['complete'],'all_numeric_rows_returned':out['all_numeric_rows_returned'],'rows':len(out['rows']),'cpu':out['cpu']}))

if __name__=='__main__':main()
