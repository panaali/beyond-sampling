"""Saved selection contractions only; selected-rule values were never evaluated."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json
P=Path(__file__).resolve().parent
p=json.loads((P/'results.json').read_text());assert p['complete']
rows=[json.loads(x) for x in (P/'bounds.jsonl').read_text().splitlines()]
assert len(rows)==p['evaluations']==400 and len({tuple(r['key'])+(r['order'],) for r in rows})==400
assert [r['index'] for r in rows]==list(range(400))
assert sum(r['origin']=='cached' for r in rows)==84 and sum(r['origin']=='new' for r in rows)==316
out=dict(results_sha256=hashlib.sha256((P/'results.json').read_bytes()).hexdigest(),bounds_sha256=hashlib.sha256((P/'bounds.jsonl').read_bytes()).hexdigest(),evaluations=400,cached=84,new=316,cpu=p['cpu'],panels={})
for panel in ('original','child'):
 rr=[r for r in p['groups'] if r['panel']==panel];assert len(rr)==11
 success=[r for r in rr if r['selection']['status']=='certified_ideal'];capped=[r['fixture'] for r in rr if r['selection']['status']=='capped'];unknown=[r['fixture'] for r in rr if r['selection']['status']=='unknown']
 out['panels'][panel]=dict(certified=len(success),capped=capped,unknown=unknown,orders={r['fixture']:r['selection']['selected_order'] for r in success},maximum_selected_order=max(r['selection']['selected_order'] for r in success),maximum_predicted_nodes=max(r['selection']['predicted_nodes'] for r in success),maximum_predicted_kernel_proxy=max(r['selection']['predicted_source_kernel_proxy'] for r in success),sum_predicted_nodes_certified_only=sum(r['selection']['predicted_nodes'] for r in success),sum_predicted_kernel_proxy_certified_only=sum(r['selection']['predicted_source_kernel_proxy'] for r in success),maximum_selected_bound=float(max(F(r['selection']['bound_sum']) for r in success)))
out['fresh_production_bound_counts']=p['fresh_production_bound_counts'];out['new_research_bound_counts']=p['new_research_bound_counts'];out['cold_counts']=p['cold_counts'];out['selected_rule_evaluations']=0
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['panels']))
