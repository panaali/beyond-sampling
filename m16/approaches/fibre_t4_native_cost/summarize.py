"""Contract saved ledgers; keep diagnostic integer counters and exclusions visible."""
import json
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent

def contract(row):
 ledger=row['ledger'];totals=Counter()
 for site,counts in ledger['sites'].items():totals.update(counts)
 assert dict(totals)==ledger['totals']
 explicit=sum(totals.values());assert explicit==ledger['inclusive_explicit_total']
 diagnostic=sum(sum(counts.values()) for site,counts in ledger['sites'].items() if "stats['traces'] += 1" in site or "stats['pieces'] += 1" in site)
 dense={site:counts for site,counts in ledger['sites'].items() if site.startswith('rows_at:')}
 dense_total=sum(v for counts in dense.values() for k,v in counts.items() if not k.startswith('integer_control'))
 dense_int=sum(v for counts in dense.values() for k,v in counts.items() if k.startswith('integer_control'))
 out={'inclusive_explicit':explicit,'named_numerical':ledger['named_numeric_total'],'explicit_integer_control':ledger['integer_control_total'],'diagnostic_stats_units_included':diagnostic,'inclusive_after_diagnostic_stats':explicit-diagnostic,'rows_at_named':dense_total,'rows_at_integer':dense_int,'opaque_calls':ledger['opaque_calls'],'cpu_seconds':row['cpu_seconds'],'scope':'Model-dependent component only; cold setup and control/memory/runtime remain excluded. Native scalar elementary functions use the declared unit convention.'}
 if 'legacy_ops' in row:out.update(legacy_ops=row['legacy_ops'],visible_to_legacy_ratio=(explicit-diagnostic)/row['legacy_ops'])
 return out

rows={}
for name in ('primitive','zero','constant0','plus3_minus0','plus0_minus3'):
 r=json.loads((HERE/('controls_'+name+'.json')).read_text());assert r['complete'];rows[name]=contract(r)
assert rows['primitive']['rows_at_named']==892 and rows['primitive']['rows_at_integer']==16
out={'controls':rows,'complete_component':False}
if (HERE/'component.json').exists():
 r=json.loads((HERE/'component.json').read_text())
 if r['complete']:
  out.update(complete_component=True,component=contract(r));c=out['component'];assert c['rows_at_integer']%4==0;calls=c['rows_at_integer']//4;assert c['rows_at_named']==2095*calls;c['rows_at_calls']=calls
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
