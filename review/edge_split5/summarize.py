"""Contract fixed saved child results without replaying any integrator."""
from pathlib import Path
from collections import Counter
import json,hashlib
P=Path(__file__).resolve().parent
p=json.loads((P/'numerical.json').read_text());g=json.loads((P/'geometry.json').read_text());assert p['complete'] and g['complete']
out=dict(numerical_sha256=hashlib.sha256((P/'numerical.json').read_bytes()).hexdigest(),geometry_sha256=hashlib.sha256((P/'geometry.json').read_bytes()).hexdigest(),groups=[],children=len(p['children']),evaluations=len(p['rows']),exceptions=[r for r in p['rows'] if 'exception' in r],geometry_cpu=g['cpu'],numerical_cpu=p['cpu'])
for n in (6,10,16):
 rr=[r for r in p['groups'] if r['order']==n];assert len(rr)==11 and all(r['complete'] for r in rr)
 worst=max(rr,key=lambda r:float(r['absolute_error']))
 out['groups'].append(dict(order=n,absolute_pass=sum(r['absolute_pass'] for r in rr),relative_pass=sum(r['relative_pass'] for r in rr),bound_pass=sum(r['bound_qualifies'] for r in rr),worst_fixture=worst['fixture'],maximum_absolute_error=worst['absolute_error'],source_proxy_ops=sum(r['source_proxy_ops'] for r in rr),nodes=sum(r['nodes'] for r in rr),assembly_additions=11))
for label,items in [('geometry_exact_counts',[r['exact_counts'] for r in g['rows']]),('radial_exact_counts',[r['certificate']['counts'] for r in p['children']]),('directed_counts',[r['ideal_rule_bound']['counts'] for r in p['rows']]),('bound_coefficient_counts',[r['ideal_rule_bound']['coefficient_counts'] for r in p['rows']])]:
 c=Counter()
 for item in items:c.update(item)
 out[label]=dict(c)
out['radial_max_bits']=max(r['certificate']['max_bits'] for r in p['children']);out['bound_cold_counts']=p['bound_cold_counts']
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['groups']))
