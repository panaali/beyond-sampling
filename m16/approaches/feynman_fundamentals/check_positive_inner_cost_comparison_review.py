"""Saved category comparison only; no scalar primitive or quadrature calls."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/positive_inner5_cost_compare')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.process_time();p=read(P/'protocol.json');rp=read(P/'run_protocol.json');r=read(P/'results.json')
for pins in (p['pins'],rp['pins']):
 for path,h in pins.items():assert sha(path)==h
old=read(p['old_results']);new=read(p['new_results']);assert r['complete'] and old['complete'] and new['complete']
ob={(x['fixture'],x['side'],x['order']):x for x in old['rows']};nb={(x['fixture'],x['side'],x['order']):x for x in new['rows'] if x['kind']=='split'}
assert len(ob)==len(nb)==len(r['rows'])==66 and set(ob)==set(nb)
refs={(x['fixture'],x['side']):F(x['reference120']) for x in old['children']};groups={};seen=set()
for row in r['rows']:
 key=row['fixture'],row['side'],row['order'];assert key not in seen;seen.add(key);a,b=ob[key],nb[key];n=key[2];M=n**4
 assert a['finite'] and a['probability_range'] and b['complete'] and b['finite_positive'] and a['value'].hex()==row['old_value_hex'] and b['value'].hex()==row['new_value_hex']
 assert F(b['reference'])==refs[key[:2]] and F(row['old_exact_reference_error'])==F(a['value'])-refs[key[:2]] and F(row['new_exact_reference_error'])==F(b['value'])-refs[key[:2]]
 expected={'arithmetic':51*M+3,'comparison_boolean':1,'abs':1,'sqrt':0,'finite_check':0,'power':6*M+1};assert row['old_vector']==expected and row['old_unit_opaque_total']==57*M+6
 cross=b['branches'].get('cross_or_endpoint',0);same=b['branches'].get('same_sign',0);assert cross+same==n**3 and set(b['branches'])<={'cross_or_endpoint','same_sign'}
 nv={'arithmetic':(110+62)*cross+(114+62)*same+5,'comparison_boolean':41*n**3+2,'abs':12*n**3+1,'sqrt':3*n**3,'finite_check':11*n**3+1,'power':0}
 assert nv==row['new_vector']=={k:b['inner_counts'].get(k,0)+b['construction_assembly_counts'].get(k,0) for k in nv}
 assert row['new_unit_opaque_total']==sum(nv.values())==239*cross+243*same+9
 assert row['old_legacy_proxy']==a['source_proxy_ops']==40*M+125 and row['opaque_det5_calls_each']==1 and b['opaque_native_calls']=={'det_float64_5x5':1}
 groups.setdefault((key[0],n),[]).append(row)
assert len(groups)==len(r['groups'])==33
for g in r['groups']:
 key=g['fixture'],g['order'];members=groups[key];assert sorted(x['side'] for x in members)==[0,1]
 prior=next(x for x in new['groups'] if (x['fixture'],x['order'],x['kind'])==(*key,'split'));ref=F(prior['reference'])
 for side,lookup in [('old',ob),('new',nb)]:
  total=Counter()
  for x in members:total.update(x[side+'_vector'])
  error=sum((F(lookup[key[0],x['side'],key[1]]['value']) for x in members),F())-ref
  assert dict(total)==g[side+'_vector'] and sum(total.values())==g[side+'_unit_opaque_total'] and F(g[side+'_exact_error'])==error
  assert g[side+'_absolute_pass']==(abs(error)<=F(1,10**10)) and g[side+'_relative_pass']==(abs(error/ref)<=F(1,10**6))
 assert F(g['ratio'])==F(g['new_unit_opaque_total'],g['old_unit_opaque_total']) and g['opaque_det5_calls_each']==2 and g['diagnostic_parent_addition_each']==1
summaries=[]
for s in r['summary']:
 n=s['order'];gs=[g for g in r['groups'] if g['order']==n];assert len(gs)==11
 for side in ('old','new'):
  total=Counter()
  for g in gs:total.update(g[side+'_vector'])
  assert dict(total)==s[side+'_vector'] and sum(total.values())==s[side+'_unit_opaque_total']
  for kind in ('absolute','relative'):assert s[side+'_'+kind+'_pass']==sum(g[side+'_'+kind+'_pass'] for g in gs)
 assert F(s['ratio'])==F(s['new_unit_opaque_total'],s['old_unit_opaque_total']) and s['old_legacy_proxy']==22*(40*n**4+125)
 assert s['old_universal_setup']=={'arithmetic':25*n**4+3*n+5*128,'power':2*n**4,'negation':128,'exp':128,'opaque_leggauss_calls':2}
 assert (s['new_universal_mapping'],s['new_executed_staged_mapping'],s['opaque_det5_calls_each'],s['parent_assembly_additions_each'],s['new_required_source_generation_calls'])==(3*n,6*n,22,11,1)
 summaries.append({k:s[k] for k in ('order','old_unit_opaque_total','new_unit_opaque_total','ratio','old_absolute_pass','new_absolute_pass','old_relative_pass','new_relative_pass')})
assert [s['order'] for s in summaries]==[6,10,16] and r['mass_model_geometry_reference_calls']==0
progress=[json.loads(x) for x in (P/'progress.jsonl').read_text().splitlines()];assert len(progress)==1+66+66+33
out=dict(complete=True,child_rows=66,parent_groups=33,summary=summaries,results_sha256=sha(P/'results.json'),review_cpu=time.process_time()-start,new_numerical_calls=0,scope='Saved forced-thin old versus analytic-inner category vectors and exact returned-value diagnostics checked. Unit power/sqrt/finite/abs categories are a declared expression model, not native/runtime bounds or sampler-MSE. Both retain opaque det5 and separate reusable/staged setup; lower-order accuracy losses retained.')
Path(__file__).with_name('positive_inner_cost_comparison_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
