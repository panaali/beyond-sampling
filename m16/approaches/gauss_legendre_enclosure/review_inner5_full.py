"""Independent saved-only reference/ledger audit; no candidate imports/calls."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D, localcontext
from collections import Counter
import json,hashlib,time,math
H=Path(__file__).resolve().parent
A=Path('/workspace/work/recover-small_h2_34/m16/approaches/positive_inner5/outer')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(s,f):
    with localcontext() as c:
        c.prec=200
        exact=D(f.numerator)/D(f.denominator)
        assert abs(D(s)-exact)<=max(abs(exact),D('1e-150'))*D('1e-148'),(s,str(exact))
def main():
    out=H/'review_inner5_full.json';assert not out.exists()
    start=time.process_time();P=json.loads((H/'review_inner5_full_protocol.json').read_text())
    for path,h in P['pins'].items():assert sha(Path(path))==h,path
    p=json.loads((A/'protocol.json').read_text());fp=json.loads((A/'full_protocol.json').read_text())
    for protocol in [p,fp]:
        for path,h in protocol['pins'].items():assert sha(Path(path))==h,path
    z=json.loads((A/'full.json').read_text());pilot=json.loads((A/'pilot.json').read_text())
    assert z['complete'] and z['all_numeric_rows_returned'] and len(z['rows'])==99 and len(z['groups'])==66
    assert z['full_protocol_sha256']==sha(A/'full_protocol.json')
    original=json.loads(Path(p['original_source']).read_text());children=json.loads(Path(p['child_source']).read_text())
    refs={'original/'+r['name']:r['reference120'] for r in original['fixtures']}
    refs.update({'split/'+r['fixture']+'/'+str(r['side']):r['reference120'] for r in children['children']})
    parents={r['name']:r for r in original['fixtures']}
    cats=['arithmetic','comparison_boolean','abs','sqrt','finite_check']
    tables={'constant':[35,32,10,1,11],'same_sign':[114,41,12,3,11],'cross_or_endpoint':[110,41,12,3,11]}
    keys=set();reuse=0;calls=0;grouped={};branches=Counter()
    for r in z['rows']:
        key=(r['id'],r['order']);assert key not in keys;keys.add(key)
        assert r['complete'] and r['finite_positive'] and math.isfinite(r['value']) and r['value']>0
        assert r['value'].hex()==r['value_hex'] and r['reference']==refs[r['id']]
        n=r['order'];assert n in [6,10,16] and r['inner_calls']==r['intended_inner_calls']==n**3
        assert sum(r['branches'].values())==n**3
        expected={k:sum(tables[b][i]*count for b,count in r['branches'].items()) for i,k in enumerate(cats)}
        assert expected==r['inner_counts']
        assert r['construction_assembly_counts']==dict(zip(cats,[62*n**3+5,2,1,0,1]))
        assert r['opaque_native_calls']=={'det_float64_5x5':1}
        err=F.from_float(r['value'])-F(r['reference']);rel=abs(err/F(r['reference']))
        for name,v in [('error',err),('absolute_error',abs(err)),('relative_error',rel)]:close(r[name],v)
        assert r['absolute_pass']==(abs(err)<=F(1,10**10)) and r['relative_pass']==(rel<=F(1,10**6))
        if r['reused_pilot']:
            reuse+=1;assert key==('original/positive5orthant',16)
            for k,v in pilot['row'].items():assert r[k]==v,k
        calls+=r['inner_calls'];branches.update(r['branches'])
        grouped.setdefault((r['kind'],r['fixture'],n),[]).append(r)
    assert keys=={(i,n) for i in p['fixture_ids'] for n in [6,10,16]}
    assert reuse==z['reused_pilot_rows']==1 and calls==175296
    stats={};gkeys=set()
    for g in z['groups']:
        key=(g['kind'],g['fixture'],g['order']);assert key not in gkeys;gkeys.add(key)
        rs=grouped[key];assert len(rs)==(1 if g['kind']=='original' else 2)
        counts=Counter()
        for r in rs:counts.update(r['inner_counts']);counts.update(r['construction_assembly_counts'])
        assert dict(counts)==g['named_counts'] and g['complete']
        assert g['opaque_det5_calls']==len(rs) and g['assembly_additions']==len(rs)-1
        assert g['inner_calls']==sum(r['inner_calls'] for r in rs)
        value=sum((F.from_float(r['value']) for r in rs),F(0));reference=parents[g['fixture']]['reference120']
        assert g['reference']==reference
        err=value-F(reference);rel=abs(err/F(reference))
        for name,v in [('exact_float_sum',value),('error',err),('absolute_error',abs(err)),('relative_error',rel)]:close(g[name],v)
        assert g['absolute_pass']==(abs(err)<=F(1,10**10)) and g['relative_pass']==(rel<=F(1,10**6))
        prior=(next(r for r in original['rows'] if r['fixture']==g['fixture'] and r['arm']=='duffy'+str(g['order'])) if g['kind']=='original' else next(r for r in children['groups'] if r['fixture']==g['fixture'] and r['order']==g['order']))
        for k,v in g['prior_4d_rule'].items():assert prior.get(k)==v
        sk=g['kind']+'/'+str(g['order']);s=stats.setdefault(sk,dict(groups=0,absolute_pass=0,relative_pass=0,max_absolute=F(0),named_total=0,opaque_det5=0))
        s['groups']+=1;s['absolute_pass']+=g['absolute_pass'];s['relative_pass']+=g['relative_pass'];s['max_absolute']=max(s['max_absolute'],abs(err));s['named_total']+=sum(counts.values());s['opaque_det5']+=len(rs)
    progress=[json.loads(l) for l in (A/'full_progress.jsonl').read_text().splitlines()]
    assert len(progress)==198
    for i,r in enumerate(z['rows']):
        assert progress[2*i]==dict(stage='attempt',id=r['id'],order=r['order'])
        assert progress[2*i+1]==dict(stage='row',row=r)
    assert z['canonical_mapping_named_arithmetic']==96 and z['actually_executed_staged_mapping_named_arithmetic']==192
    for s in stats.values():s['max_absolute']=str(float(s['max_absolute']))
    result=dict(passed=True,rows=99,groups=66,raw_progress=198,inner_calls=calls,new_inner_calls=calls-4096,reused_pilot=1,branches=dict(branches),stats=stats,cpu_seconds=time.process_time()-start,scope='Saved values/reference linkage and named-source ledger only; no candidate, determinant, quadrature or network replay. Det5 opaque; no new NumPy binary pin in author outer protocol. Exact rational parent sums are diagnostics, not production floating assembly or full native cost.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'review_inner5_full_failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
