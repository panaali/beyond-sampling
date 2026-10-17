"""Independently compose all trace site ledgers and separately opaque call counts."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,time,math
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h4_4/m16/approaches/plackett_cost_review');summary=json.loads((src/'summary.json').read_text());mass=json.loads((src/'mass_summary.json').read_text());rows=[json.loads((src/f'geometry_{i}.json').read_text()) for i in range(8)];allcats=Counter();opaque=Counter()
for i,r in enumerate(rows):
 assert (r['first'],r['last'])==(i*12,(i+1)*12);cats=Counter()
 for v in r['native']['sites'].values():cats.update(v)
 assert dict(cats)==r['native']['totals'] and sum(cats.values())==r['native']['inclusive_explicit_total'];allcats.update(cats)
 for v in r['native']['opaque_calls']:opaque[tuple(v['key'])]+=v['count']
 assert r['cold_setup']==rows[0]['cold_setup'] and r['rule_setup']==rows[0]['rule_setup']
allcats['scalar_Sub']-=7;assert dict(allcats)==summary['geometry_categories'];geo=sum(allcats.values());assert geo==42396504
for name in ('cold_setup','rule_setup'):
 r=rows[0][name];assert sum(r['totals'].values())==r['inclusive_explicit_total']
 for v in r['opaque_calls']:opaque[tuple(v['key'])]+=v['count']
for v in mass['opaque']:opaque[tuple(v['key'])]+=v['count']
assert opaque==Counter({tuple(v['key']):v['count'] for v in summary['opaque_calls']});visible=geo+mass['inclusive_explicit']+rows[0]['cold_setup']['inclusive_explicit_total']+rows[0]['rule_setup']['inclusive_explicit_total'];assert visible==553400768
assert sum(r['mass_callbacks'] for r in rows)==35098 and sum(r['stats']['cells'] for r in rows)==8455 and sum(r['legacy_geometry'] for r in rows)==35650854
ref=json.loads(Path('/workspace/work/recover-small_h2_910/m16/approaches/plackett_rule_doubling/summary.json').read_text());p=ref['reference_accuracy'];b=ref['conditional_reference_integration_bound'];delta=abs(ref['candidate_accuracy']-p);low=512*(p+b)*(1-p-b)/(delta+b)**2;assert math.isclose(low,summary['conditional_single_case_break_even_total_units_interval'][0],rel_tol=1e-14)
out=dict(ordered_callbacks=35098,terminal_cells=8455,geometry_explicit=geo,visible_with_setup=visible,opaque_counts={name:sum(v for k,v in opaque.items() if k[1]==name) for name in ('np.linalg.inv','np.linalg.det','np.linalg.matrix_rank','np.polynomial.legendre.leggauss')},conditional_single_case_break_even_lower=low,review_cpu=time.process_time()-start,scope='All saved site aggregates verified, source ordered-ray assertion contract previously reviewed. Opaque and runtime work unpriced; headroom is conditional algebra, not complete-cost or sampling-win claim.')
(here/'geometry_cost_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
