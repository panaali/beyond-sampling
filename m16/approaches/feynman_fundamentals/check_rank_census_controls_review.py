"""Independent saved expanded-native-observer and geometry control parity."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import json,time
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h4_4/m16/approaches');C=P/'rank_runtime_census';D=json.loads((C/'controls.json').read_text());B=json.loads((P/'rank_cost_review/baseline.json').read_text());DB=json.loads((P/'rank_cost_review/dense_baseline.json').read_text());assert len(D['rows'])==114 and len(D['dense'])==3
for observed,base in zip(D['rows']+D['dense'],B['rows']+DB['rows']):
 assert observed['output']==base;o=observed['observer'];assert o['overflow']==0 and all(not r['failures'] for r in o['kernels']);assert o['gesdd'][0]==o['gesdd'][1]==o['gesdd'][8]==6 and o['gesdd'][2]==o['gesdd'][3]==3;assert sum(r['calls'] for r in o['kernels'] if r['kernel'] in ('dlasq5','dlasq6'))==o['nested'][8];assert not any(o['nested'][i] for i in (3,7,11))
b=json.loads((C/'geometry_baseline.json').read_text());o=json.loads((C/'geometry_observed.json').read_text());assert len(b['rows'])==len(o['rows'])==4;rows=[]
for x,y in zip(b['rows'],o['rows']):
 assert x['name']==y['name'] and x['rays']==y['rays'];a,z=x['result'],y['result'];assert all(a[k]==z[k] for k in ('mode','first','last','legacy_geometry','stats','mass_callbacks'));assert z['mass_callbacks']==len(y['rays']);n=z['native'];assert not n['overflow'] and all(not r['failures'] for r in n['kernels']);assert sum(r['calls'] for r in n['kernels'] if r['kernel'] in ('dlasq5','dlasq6'))==n['nested'][8];assert y['name']=='zero' or n['gesdd'][0]>0
 rows.append(dict(name=y['name'],mode=z['mode'],mass_callbacks=z['mass_callbacks'],legacy_geometry=z['legacy_geometry'],gesdd_calls=n['gesdd'][0]))
result=dict(analytic_parity_cases=117,toy_geometry=rows,review_cpu=time.process_time()-start,scope='Expanded observer pass-through/source control gate only; no trained rerun or universal cost/convergence claim. Nested parent/child costs must not be double-counted.')
(H/'rank_census_controls_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
