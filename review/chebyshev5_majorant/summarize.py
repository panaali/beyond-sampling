"""Contract only saved majorant bounds; no numerical integration."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json
P=Path(__file__).resolve().parent
p=json.loads((P/'results.json').read_text());assert p['complete']
out=dict(results_sha256=hashlib.sha256((P/'results.json').read_bytes()).hexdigest(),rows=len(p['rows']),unresolved=sum(r['status']!='completed' for r in p['rows']),groups=[],cpu=p['cpu'],max_rss_kib=p['max_rss_kib'],cold_counts=p['cold_counts'])
counts=Counter()
for r in p['rows']:
 if r['status']=='completed':counts.update(r['counts'])
out['directed_counts']=dict(counts)
for panel in ('original','child'):
 for n in (6,10,16):
  rr=[r for r in p['groups'] if r['panel']==panel and r['order']==n];assert len(rr)==11 and all(r['complete'] for r in rr)
  worst=max(rr,key=lambda x:F(x['bound_sum_fraction']))
  out['groups'].append(dict(panel=panel,order=n,qualifies=sum(r['qualifies'] for r in rr),strictly_tighter=sum(r['strictly_tighter'] for r in rr),worst_fixture=worst['fixture'],maximum_upper=float(F(worst['bound_sum_fraction'])),maximum_upper_exact=worst['bound_sum_fraction']))
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['groups']))
