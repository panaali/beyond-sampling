import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
import ast,hashlib,json,time
import numpy as np
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/fibre_t4_recovery');inv=json.loads((P/'inventory.json').read_text());src=Path(inv['source_root']);start=time.process_time()
for x in inv['files']+inv['references']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
raw=json.loads((src/'results_H8_T4.json').read_text())['rows'];saved=json.loads((P/'contraction.json').read_text());rows={r['name']:r for r in saved['rows']}
with localcontext() as ctx:
 ctx.prec=65;ratios=[];bias=[];costs=[];variances=[]
 for name,r in raw.items():
  p=D(r['truth']);v=p*(1-p);e=D(r['F2L3']['est'])-p;se=D(r['truth_se']);cost=D(r['F2L3']['ops'])
  ratio=max(e*e-se*se,D(0))*cost/(320*v);ratios.append(float(ratio));bias.append(e*e-se*se);costs.append(cost);variances.append(v)
  assert abs(float(ratio)-rows[name]['legacy_mc_variance_subtracted_ratio'])<1e-16
  assert abs(float(v/se**2)/1e8-1)<1e-12
  if 'corrected_reference' in rows[name]:
   cr=rows[name]['corrected_reference'];assert float(D(r['F2L3']['est'])-D(cr['value']))==cr['signed_error']
 group=max(sum(bias)/5,D(0))*(sum(costs)/5)/(320*sum(variances)/5)
 assert float(group)==saved['legacy_group_ratio']
# Rank probabilities by exact polynomial antiderivative, independent of numerical CDF.
def prim(u):return u**3-F(3,4)*u**4
lastplus=prim(F(1))-prim(F(1,2));lastminus=prim(F(1,2));otherplus=(F(1,2)-lastplus)/3;otherminus=(F(1,2)-lastminus)/3
assert (lastplus,lastminus,otherplus,otherminus)==(F(11,64),F(5,64),F(7,64),F(9,64))
truths={'zero':F(1,4),'constant0':F(1,4),'plus3_minus0':lastplus+otherminus,'plus0_minus3':otherplus+lastminus}
for name,truth in truths.items():
 old=json.loads((P/('analytic_'+name+'.json')).read_text());new=json.loads((P/('guard_'+name+'.json')).read_text());assert old['expected']==float(truth) and new['complete']
 assert old['arms'][0]['estimate']==old['arms'][1]['estimate'];assert new['arms'][0]['estimate']==new['arms'][1]['estimate']
 assert all(abs(a['estimate']-float(truth))<1e-7 for a in new['arms'])
 if name!='zero':assert [a['estimate'] for a in old['arms']]==[a['estimate'] for a in new['arms']]
 else:assert all(a['estimate']==1 and a['stats']['traces']==0 for a in old['arms'])
# Execute only pinned clip AST on a single triangle, no estimator or model.
tree=ast.parse((src/'fibre4.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='clip');ns={'np':np,'EPS':1e-10};exec(compile(ast.Module(body=[node],type_ignores=[]),'pinned clip only','exec'),ns)
poly=list(np.eye(3))
with np.errstate(invalid='ignore',divide='ignore'):bad=ns['clip'](poly,np.zeros(3))
assert bad==([],[])
# Exact zero normal is the vacuous inequality; retain once, avoiding duplicate mass.
assert all(sum(F(0)*F(float(v)) for v in ray)==0 for ray in poly)
out=dict(clean_rows=len(raw),legacy_ratios=ratios,legacy_group_ratio=float(group),rank_sign_masses=list(map(str,[lastplus,lastminus,otherplus,otherminus])),original_zero_clip_discarded_both=True,guarded_four_case_parity=True,review_cpu_seconds=time.process_time()-start,scope='Saved arithmetic and exact rank/zero-halfspace reasoning. No trained rerun, new cost convention, noise pass, or original checkpoint execution provenance established.')
Path(__file__).with_name('fibre_t4_recovery_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
