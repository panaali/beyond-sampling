"""Saved helper controls and original rank-census driver contraction."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,math,struct
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope/driver_review')
R=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(P/'helper_controls.json');proto=read(P/'helper_protocol.json');b=read(P/'contraction.json')
assert sha(P/'helper_controls.py')==proto['source_sha256'] and sha(R/'summary.json')==proto['summary_sha256']
assert len(a['rows'])==a['cases']==439 and a['failures']==0
types=Counter();shapes=set()
for x in a['rows']:
    assert x['match'] and x['count']<=x['upper'];types[x['args']['kind']]+=1
    assert len(x['expected'])==len(x['actual'])
    for e,v in zip(x['expected'],x['actual']):
        assert e==v or (math.isnan(float.fromhex(e)) and math.isnan(float.fromhex(v)))
    k=x['args']['kind']
    if k=='rotation':assert x['upper']==25
    elif k=='maxnorm':assert x['upper']==3*x['args']['m']*x['args']['n']
    elif k=='roundup':assert x['upper']==2
    else:
        assert k=='blocksize' and x['extra']['blocks_absent']
        assert float.fromhex(x['actual'][0])==32>=min(x['args']['m'],x['args']['n'])
        shapes.add((x['args']['name'],x['args']['m'],x['args']['n']))
assert dict(types)==dict(rotation=100,maxnorm=288,roundup=10,blocksize=41)
assert shapes==set(map(tuple,proto['environment_shapes']))
def census(p):
    assert p['overflow']==0
    c=Counter()
    for r in p['kernels']:
        if r['kernel']=='dgesdd':
            assert r['failures']==0 and min(r['m'],r['n'])<=4 and max(r['m'],r['n'])<=24
            c[r['m'],r['n'],r['query']]+=r['calls']
        if r['kernel'] in ('dgeqrf','dgelqf','dgebrd'):assert (r['kernel'],r['m'],r['n']) in shapes
    return c
totals={}
for mode in ('exact','recipe108'):
    geo=Counter();cold=None
    for j in range(8):
        path=R/f'{mode}_{j}.json';assert sha(path)==b['pins'][path.name];v=read(path)
        geo+=census(v['native']);cc=census(v['cold'])
        if cold is None:cold=cc
        else:assert cold==cc
    for phase,c in [('geometry',geo),('cold_once',cold),('total',geo+cold)]:
        rows=b['modes'][mode][phase]['rows'];assert len(rows)==len(c)
        assert {(r['m'],r['n'],r['query']):r['calls'] for r in rows}==dict(c)
        total=0
        for r in rows:
            unit=4 if r['query'] else 194+6*r['m']*r['n']+3*min(r['m'],r['n'])
            assert r['per_call_source_upper']==unit and r['source_upper']==unit*r['calls'];total+=unit*r['calls']
        assert total==b['modes'][mode][phase]['source_upper']
        assert sum(c.values())==b['modes'][mode][phase]['calls']
    totals[mode]=b['modes'][mode]['total']['source_upper']
composition=read(P.parent/'composition.json')
for path,h in composition['pins'].items():assert sha(P.parent/path)==h
combined={}
for mode,z in composition['modes'].items():
    assert z['disjoint_other_components']['driver']==totals[mode]
    other=sum(z['disjoint_other_components'].values())
    assert other+z['reflector_norm_helpers_conditional']==z['conditional_named_subset']
    assert other+z['reflector_norm_helpers_generic']==z['generic_named_subset']
    combined[mode]=dict(conditional=z['conditional_named_subset'],generic=z['generic_named_subset'])
out=dict(complete=True,helper_controls=dict(types),saved_driver_shapes_verified=41,driver_source_totals=totals,
         combined_qualified_source_categories=combined,
         controls_sha256=sha(P/'helper_controls.json'),contraction_sha256=sha(P/'contraction.json'),
         new_native_rank_or_model_calls=0,
         scope='Successful finite values-only min-dimension<=4 source-category upper. Cold counted once; generic scaling/rotation allowances retained. Previously priced unblocked/Q1 descendants excluded. Not all-input compiled-lane equivalence or full runtime cost.')
(HERE/'rank_driver_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
