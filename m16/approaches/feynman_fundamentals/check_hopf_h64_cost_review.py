import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_native_cost');ROOT=Path('/workspace/work/recover-small_h2_78/m16/approaches/bulk_hopf_review');start=time.process_time()
def read(p):return json.loads(p.read_text())
def ledger(l):
 c=Counter()
 for row in l['sites'].values():c.update(row)
 assert c==Counter(l['totals']);integer=sum(v for k,v in c.items() if k.startswith('integer_control_'));assert integer==l['integer_control_total'] and sum(c.values())-integer==l['named_numeric_total'] and sum(c.values())==l['inclusive_explicit_total'];return c
rows=[]
for arm in ('original','tile'):
 d=read(P/f'model_h64_{arm}.json');assert d['status']=='complete' and d['arm']==arm and (d['H'],d['T'],d['seed'],d['nu'],d['nd'])==(4,4,0,64,128)
 for name,h in d['pins'].items():
  p=ROOT/'frozen/ev_44.json' if name=='frozen_ev_44.json' else P/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 assert d['model_sha256']==read(P/'model_identity.json')['float64_c_order_sha256'];saved=read(ROOT/'frozen/ev_44.json')['0']['rows'][0];assert d['value']==d['baseline_value']==d['saved_original_value']==saved['h64'];assert d['legacy']==d['baseline_legacy']==d['saved_original_legacy']==saved['h64_ops']
 ledger(d['ledger']);ledger(d['startup']);l=d['ledger'];by=Counter()
 for r in l['source_closures']:
  if r['name']=='np.lexsort':assert r['numerical_allowance']==r['source_scalar_comparisons'] and r['source_comparator_calls']<=r['source_scalar_comparisons']<=3*r['source_comparator_calls']
  elif r['name']=='np.searchsorted':assert r['source_comparator_calls']==r['keys']*(max((r['n']-1).bit_length(),1)+1) if r['n'] else r['source_comparator_calls']==0;assert r['numerical_allowance']==4*r['source_comparator_calls']
  else:assert r['name']=='np.argsort' and r['n']==4 and r['numerical_allowance']==157*r['rows']
  by[r['name']]+=r['numerical_allowance']
 assert sum(by.values())==l['source_closure_allowance'];assert len(l['source_closures'])==sum(x['count'] for x in l['opaque_calls']);assert d['explicit_with_constant_setup']==l['inclusive_explicit_total']+d['startup']['inclusive_explicit_total']==148561137;assert d['conditional_source_with_constant_setup']==d['explicit_with_constant_setup']+sum(by.values())==252465096
 assert by==Counter({'np.lexsort':8629307,'np.searchsorted':2750784,'np.argsort':92523868});rows.append(d)
assert rows[0]['ledger']['totals']==rows[1]['ledger']['totals']
out=dict(arms=2,value=rows[0]['value'],legacy=65934700,inclusive_explicit_with_setup=148561137,conditional_source_with_setup=252465096,source_allowance_by_call=dict(by),review_cpu=time.process_time()-start,scope='Pinned single H4T4 seed0 h64 two-arm saved outputs and complete site/allowance sums. This model has no observed tile change. Source-shadow/native-dispatch, mixed-category and full-cost exclusions persist; no panel or sampling inference.')
Path(__file__).with_name('hopf_h64_cost_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
