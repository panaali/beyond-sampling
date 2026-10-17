"""Independent primitive site sums and branch contraction; opaque kernels separate."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,time
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h4_4/m16/approaches/plackett_cost_review');d=json.loads((src/'controls.json').read_text());mass=json.loads((src/'mass_summary.json').read_text());expected=dict(wide=4808,thin=18632,cutoff=184);categories=Counter()
for row in d['rows']:
 ledger=row['ledger'];summed=Counter()
 for site,count in ledger['sites'].items():summed.update(count)
 assert dict(summed)==ledger['totals'];assert sum(summed.values())==ledger['inclusive_explicit_total']==expected[row['name']]
 categories.update({k:v*mass['counts'][row['name']] for k,v in summed.items()})
assert dict(categories)==mass['categories'];assert sum(categories.values())==mass['inclusive_explicit']==510972352
nodes=24*10245+96*24780;assert nodes==2624760;assert 200*(10245+24780)+184*73+192*nodes==510972352
assert 156*35098+152*nodes==mass['legacy_partial']==404438808
sample=d['sampler'];assert sum(sample['totals'].values())==512==16+3*144+60+4
opaque={r['key'][0]:r['count'] for r in mass['opaque']};assert sum(v for k,v in opaque.items() if 'inv' in k)==35098+nodes;assert sum(v for k,v in opaque.items() if 'det' in k)==35098
out=dict(primitive_counts=expected,mass_explicit=510972352,legacy_partial=404438808,sampler=512,inverse_calls=35098+nodes,determinant_calls=35098,review_cpu=time.process_time()-start,scope='Source-expression convention; inverse/det opaque, memory/index and implicit runtime work excluded. Geometry source reviewed with ordered-ray parity contract; no complete-cost conclusion.')
(here/'plackett_cost_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
