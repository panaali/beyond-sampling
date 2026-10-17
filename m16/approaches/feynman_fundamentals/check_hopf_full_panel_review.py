"""Audit every frozen panel site and closure after all25 records exist."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,time
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_native_cost')
R=Path('/workspace/work/recover-small_h2_78/m16/approaches/bulk_hopf_review')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time()
models=read(P/'panel_models.json'); summary=read(P/'panel_summary.json')
saved=read(R/'frozen/ev_44.json'); previous=read(HERE/'hopf_panel_pilot_review_results.json')
assert previous['reconstructed_saved_models']==25
assert len(models['rows'])==len(summary['rows'])==25
assert {(r['seed'],r['copy']) for r in models['rows']}=={(s,c) for s in range(5) for c in range(5)}
reported={(r['seed'],r['copy']):r for r in summary['rows']}; allrows=[]
for m in models['rows']:
    key=(m['seed'],m['copy']); s,c=key
    path=P/('model_h64_original.json.gz' if key==(0,0) else f'panel_s{s}_c{c}.json.gz')
    raw=gzip.decompress(path.read_bytes());d=json.loads(raw);rr=reported[key]
    assert rr['compressed_sha256']==sha(path) and rr['raw_sha256']==hashlib.sha256(raw).hexdigest()
    assert d['status']=='complete' and d['arm']=='original' and (d['nu'],d['nd'])==(64,128)
    assert d['model_sha256']==m['model_sha256']==rr['model_sha256']
    old=saved[str(s)]['rows'][c]
    assert d['value']==d['baseline_value']==old['h64']==rr['value']
    assert d['legacy']==d['baseline_legacy']==old['h64_ops']==rr['legacy']
    for name,h in d['pins'].items():assert sha(R/'frozen/ev_44.json' if name=='frozen_ev_44.json' else P/name)==h
    for l in (d['ledger'],d['startup']):
        cc=Counter()
        for cats in l['sites'].values():cc.update(cats)
        assert cc==Counter(l['totals'])
        ints=sum(v for k,v in cc.items() if k.startswith('integer_control'))
        assert ints==l['integer_control_total'] and sum(cc.values())-ints==l['named_numeric_total']
        assert sum(cc.values())==l['inclusive_explicit_total'] and not l['complete_implementation_cost']
    l=d['ledger']; by=Counter()
    for cl in l['source_closures']:
        name=cl['name']
        if name=='np.lexsort':
            assert cl['numerical_allowance']==cl['source_scalar_comparisons']
            assert cl['source_comparator_calls']<=cl['source_scalar_comparisons']<=3*cl['source_comparator_calls']
        elif name=='np.searchsorted':
            n=cl['n'];expected=cl['keys']*(max((n-1).bit_length(),1)+1) if n else 0
            assert cl['source_comparator_calls']==expected and cl['numerical_allowance']==4*expected
        else:
            assert name=='np.argsort' and cl['n']==4 and cl['numerical_allowance']==157*cl['rows']
        by[name]+=cl['numerical_allowance']
    assert sum(by.values())==l['source_closure_allowance'] and dict(by)==rr['closures']
    assert len(l['source_closures'])==sum(v['count'] for v in l['opaque_calls'])==rr['opaque_calls']
    explicit=l['inclusive_explicit_total']+d['startup']['inclusive_explicit_total']
    assert explicit==d['explicit_with_constant_setup']==rr['explicit']
    assert explicit+sum(by.values())==d['conditional_source_with_constant_setup']==rr['conditional']
    assert rr['reused_prior_record']==(key in ((0,0),(0,1)))
    allrows.append(dict(seed=s,copy=c,explicit=explicit,conditional=explicit+sum(by.values()),legacy=d['legacy'],opaque_calls=len(l['source_closures'])))
for k in ('legacy','explicit','conditional','opaque_calls'):
    assert sum(r[k] for r in allrows)==summary['totals'][k]
out=dict(complete=True,rows=allrows,totals={k:summary['totals'][k] for k in ('legacy','explicit','conditional','opaque_calls')},
         summary_sha256=sha(P/'panel_summary.json'),review_cpu=time.process_time()-start,
         scope='All25 frozen original-source records and complete closure/site ledgers. Two prior records reused once. Conditional source allowances remain incomplete native/runtime cost; original root-mask defect, finite-reference and statistical limits retained.',new_model_mass_reference_calls=0)
(HERE/'hopf_full_panel_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
