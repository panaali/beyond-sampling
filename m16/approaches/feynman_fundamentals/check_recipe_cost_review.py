"""Independent recipe expression aggregation and determinant-only branch contraction."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,time,math,hashlib
import numpy as np
start=time.process_time();here=Path(__file__).parent;root=Path('/workspace/work/recover-small_h4_4/m16/approaches');src=root/'recipe_cost_review';prior=root/'plackett_cost_review';summary=json.loads((src/'summary.json').read_text());mass=json.loads((src/'mass_census.json').read_text());cats=Counter();functions=Counter();opaque=Counter();rows=[]
for b in range(8):
 r=json.loads((src/f'geometry_{b}.json').read_text());rows.append(r);assert (r['first'],r['last'])==(12*b,12*b+12);assert hashlib.sha256((src/'run.py').read_bytes()).hexdigest()==r['adapter_sha256'];local=Counter()
 for site,v in r['native']['sites'].items():local.update(v);functions[site.split(':')[0]]+=sum(v.values())
 assert dict(local)==r['native']['totals'];cats.update(local)
 for v in r['native']['opaque_calls']:opaque[tuple(v['key'])]+=v['count']
 assert r['cold_setup']==rows[0]['cold_setup'] and r['rule_setup']==rows[0]['rule_setup']
cats['scalar_Sub']-=7;functions['estT']-=7;assert dict(cats)==summary['geometry_categories'];assert dict(functions)==summary['geometry_function_totals'];geo=sum(cats.values());assert geo==45131971
census=np.load(src/'det_census.npz');cut=census['cutoff'];assert np.array_equal(cut,abs(census['determinants'])<1e-12);assert int(cut.sum())==54;N=len(cut);nodes=(325162176-156*N)//152;thin=(nodes//24-(N-54))//3;wide=N-54-thin;assert (nodes,thin,wide)==(2110488,19997,7949)
branches=json.loads((prior/'controls.json').read_text())['rows'];mcats=Counter()
for row in branches:
 for k,v in row['ledger']['totals'].items():mcats[k]+=v*mass['counts'][row['name']]
assert dict(mcats)==mass['categories'];assert sum(mcats.values())==200*N+192*nodes-16*54==410812832
for ledger,field in [(mass,'opaque'),(rows[0]['cold_setup'],'opaque_calls'),(rows[0]['rule_setup'],'opaque_calls')]:
 for v in ledger[field]:opaque[tuple(v['key'])]+=v['count']
assert opaque==Counter({tuple(v['key']):v['count'] for v in summary['opaque_calls']});visible=geo+sum(mcats.values())+sum(rows[0][name]['inclusive_explicit_total'] for name in ('cold_setup','rule_setup'));assert visible==455976715
assert sum(r['mass_callbacks'] for r in rows)==28000 and sum(r['stats']['cells'] for r in rows)==6182 and sum(r['legacy_geometry'] for r in rows)==36398526
p=summary['reference_accuracy'];err=abs(summary['candidate_accuracy']-p);bound=summary['conditional_reference_integration_bound'];low=512*(p+bound)*(1-p-bound)/(err+bound)**2;assert math.isclose(low,summary['conditional_single_case_break_even_total_units_interval'][0],rel_tol=1e-14)
out=dict(cones=N,cells=6182,geometry_explicit=geo,mass_explicit=410812832,total_visible=visible,opaque_counts={name:sum(v for k,v in opaque.items() if k[1]==name) for name in ('np.linalg.inv','np.linalg.det','np.linalg.matrix_rank','np.polynomial.legendre.leggauss')},cutoffs=54,wide=wide,thin=thin,nodes=nodes,conditional_break_even_lower=low,review_cpu=time.process_time()-start,scope='Pinned recipe source/ordered-ray parity contract and all site aggregates. Branch counts derived from unchanged per-node counter plus source-identical determinant census. Opaque/runtime work remains unpriced; no full-cost win or new candidate selection.')
(here/'recipe_cost_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
