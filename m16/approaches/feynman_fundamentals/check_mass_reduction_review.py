"""Saved finite control, metadata and replacement-only contraction review."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import hashlib,json,math
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/mass_reduction_cost')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a,b):return a==b or (math.isnan(float.fromhex(a)) and math.isnan(float.fromhex(b)))
protocol=read(P/'protocol.json');d=read(P/'controls.json');s=read(P/'summary.json')
assert protocol['controls_sha256']==sha(P/'controls.py')
assert d['cases']==172 and d['failures']==[] and len(d['rows'])==172
assert [json.loads(x) for x in (P/'control_progress.jsonl').read_text().splitlines()]==d['rows']
counts=Counter(r['kind'] for r in d['rows']);assert counts==Counter(numpy_sum=60,norm4=40,python_sum6=72)
for r in d['rows']:
    assert r['ok']
    if r['kind']=='numpy_sum':
        assert len(r['values'])==len(r['reference']) and all(same(a,b) for a,b in zip(r['values'],r['reference']))
        shape=r['shape'];n=shape[-1];rows=1 if len(shape)==1 else shape[0]
        assert r['count']==rows*(n+1 if n<8 else n)
    else:
        assert same(r['value'],r['reference'])
        if r['kind']=='norm4' and all(math.isfinite(float.fromhex(x)) for x in r['input']):
            acc=0.0
            for x in map(float.fromhex,r['input']):
                if math.isinf(acc):break
                try:acc=float(F(acc)+F(x)*F(x))
                except OverflowError:acc=math.inf
            assert same(r['reference'],math.sqrt(acc).hex())
for file,h in s['pins'].items():assert sha(P/file)==h
for manifest in ('manifest.json','source_extra.json'):
    for name,pin in read(P/manifest).items():assert sha(P/'frozen'/name)==pin['sha256']
records=[json.loads(x) for x in (P/'dispatch_offset0.jsonl').read_text().splitlines()]
assert records[-1]['stage']=='complete'
maps=[r for r in records if r['stage']=='library_maps']
bases={r['path']:next(m[0] for m in r['rows'] if m[3]==0) for r in maps}
for r in records:
    if r['stage']=='read_check':assert r['mapped']
    if r['stage']=='info':assert r['loop_offset']=='0x2ebee0'
    if r['stage']=='aux':assert r['legacy_candidate_offset']=='0x5462e0'
    if r['stage']=='dot':assert r['offset']=='0xa3d7c0'
for path,h in records[-1]['binary_pins'].items():assert sha(Path(path))==h
prior=read(P/'frozen/prior_summary.json');census=read(P/'frozen/full_summary.json')
assert (census['thin'],census['wide'],census['cases'])==(17955,17143,35098)
expected={'thin_row4':('thin',1000,3,4),'thin_sum1000':('thin',1,999,1000),
          'wide_norm4':('wide',16,8,10),'wide_sum128':('wide',6,127,128),'wide_python_sum6':('wide',1,5,61)}
removed=replacement=0
for name,(branch,mult,old,new) in expected.items():
    r=s['sites'][name];copies=census[branch]
    assert (r['branch'],r['multiplicity'],r['old_per'],r['new_per'],r['copies'])==(branch,mult,old,new,copies)
    a,b=copies*mult*old,copies*mult*new
    assert (r['removed'],r['replacement'],r['delta'])==(a,b,b-a)
    removed+=a;replacement+=b
assert s['removed_once']==removed and s['replacement']==replacement
assert s['delta']==replacement-removed==17955*1001+17143*94==19584397
assert s['prior_partial_hybrid']==prior['revised_hybrid_subtotal']==3572147383
assert s['revised_partial_hybrid']==3591731780
assert s['model_free_constants_unchanged']==190197 and not s['complete_native_cost'] and s['sampling_ratio_not_computed']
out=dict(complete=True,controls=dict(counts),current_metadata_offsets=['0x2ebee0','0x5462e0','0xa3d7c0'],
         replacement_delta=s['delta'],revised_partial_hybrid=s['revised_partial_hybrid'],constants_separate=190197,
         summary_sha256=sha(P/'summary.json'),new_kernel_mass_calls=0,
         scope='Saved controls and current metadata with reviewed selected binary paths; replacement-only category arithmetic. Historical row traversal/binding remains conditional. Source short-row count5 differs from compiled count4; partial hybrid is not full native cost, hardware cost or a sampling comparison.')
(HERE/'mass_reduction_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
