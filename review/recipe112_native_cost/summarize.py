"""Named source-expression totals, preserving explicit exclusions."""
import json
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
out=dict(scope='Two fixed native-source diagnostics only; no scalar-cost multiplier transfer or full implementation ceiling.',rows=[],review_source_gate='ea2609ef')
for H in (4,8):
 d=json.loads((HERE/f'trained_H{H}.json').read_text());ledger=d['traced']['ledger'];by=defaultdict(int)
 for site,counts in ledger['sites'].items():
  for key,n in counts.items():
   if not key.startswith('integer_control'):by[site.split(':')[0]]+=n
 n=sum(by.values());assert n==ledger['named_numeric_total']
 out['rows'].append(dict(H=H,native_named=n,integer_control=ledger['integer_control_total'],scalar=d['scalar_repaired_counter'],legacy=d['baseline']['legacy'],native_to_scalar=n/d['scalar_repaired_counter'],native_to_legacy=n/d['baseline']['legacy'],by_function=dict(sorted(by.items(),key=lambda x:-x[1])),memory_calls=sum(r['count'] for r in ledger['memory_calls']),startup=d['traced']['startup'],cpu=d['cpu']))
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['rows'],indent=2))
