import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib,time
start=time.process_time();P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_native_cost');F=Path('/workspace/work/recover-small_h2_910/review/fibre_native_completion')
def read(p):return json.loads(Path(p).read_text())
def ledger(l):
 c=Counter()
 for v in l['sites'].values():c.update(v)
 assert c==Counter(l['totals']);integer=sum(v for k,v in c.items() if k.startswith('integer_control_'));numeric=sum(c.values())-integer
 assert integer==l['integer_control_total'] and numeric==l['named_numeric_total'] and sum(c.values())==l['inclusive_explicit_total'];assert l['complete_implementation_cost'] is False
 return c
h=read(P/'controls.json');assert h['status']=='complete' and len(h['rows'])==15;assert hashlib.sha256((P/'tracer.py').read_bytes()).hexdigest()==h['source_sha256'];w=read('/workspace/work/recover-small_h2_78/m16/approaches/bulk_hopf_review/full_circle_control.json');groups=Counter()
for r in h['rows']:
 ledger(r['ledger']);groups[r['case']]+=1
 if r['case']=='circle' and r['model']==0:assert r['value'][0]==w['tile_rescue_fraction' if r['rescue'] else 'original_fraction']
 if r['case']=='base_grid':assert r['ledger']['inclusive_explicit_total']==34246 and r['legacy']==15396;assert sum(x['count'] for x in r['ledger']['opaque_calls'])==42
sample=h['rows'][-1];c=ledger(sample['ledger']);assert sample['fixed_inputs']==5 and sample['draws']==0;assert c['matrix_mul']==400 and c['matrix_add']==300 and c['array_Mult']==80 and c['array_Add']==80 and c['array_compare']==85 and c['reduce_argmax']==15 and c['reduce_.sum']==3
# 960 indicator expressions, three within-batch hit additions, two scalar-int hit accumulations and final divide; other integer loop/index work distinct.
assert sample['ledger']['named_numeric_total']==5*192+3
cold=read(F/'cold.json');assert cold['complete'];total=Counter();opaque=Counter();targets=0
for name,r in cold['modules'].items():
 ledger(r['ledger']);total.update(r['ledger']['totals']);assert all(r['targets'].values());targets+=len(r['targets'])
 for x in r['ledger']['opaque_calls']:opaque[x['key'][1]]+=x['count']
assert targets==25 and opaque==Counter({'np.linalg.svd':6,'np.linalg.inv':48,'np.polynomial.legendre.leggauss':6});assert sum(v for k,v in total.items() if k.startswith('integer_control_'))==16 and sum(total.values())==6930
assembly=read(F/'assembly.json');c=ledger(assembly['ledger']);assert assembly['complete'] and assembly['indices']==list(range(48));assert c==Counter(scalar_Add=48,scalar_Div=1,scalar_Sub=1);assert assembly['estimate'].hex()==read(Path(read(F/'inventory.json')['prior'])/'noise_summary.json')['estimate'].hex()
for name,d in [('cold',cold),('assembly',assembly)]:assert hashlib.sha256((F/f'{name}.py').read_bytes()).hexdigest()==d['runner_sha256']
out=dict(hopf_control_cases=15,control_groups=dict(groups),sampler_named=963,sampler_integer=20,cold_targets=targets,cold_inclusive=6930,cold_integer=16,cold_opaque=dict(opaque),outer_assembly_numeric=50,review_cpu=time.process_time()-start,scope='Independent saved site/category sums and source-linked finite controls. Hopf opaque sort/search and native interiors remain open. Sampler integer hit-aggregation tags are explicit; no automatic denominator transfer. Fibre cold imports/libraries remain excluded and components counted separately.')
Path(__file__).with_name('native_cold_hopf_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
