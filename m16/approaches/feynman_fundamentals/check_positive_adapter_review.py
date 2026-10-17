import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import ast,hashlib,importlib.util,json,marshal,math,time,types,sys,numpy as np
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/positive_model_adapter');start=time.process_time();spec=importlib.util.spec_from_file_location('review_adapter',P/'adapter.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
old=np.load;attempts=[]
def reject(*a,**k):attempts.append(str(a[0]));raise AssertionError('Stored array loaded')
np.load=reject
try:E,counts,M,positive,kernel,rescue,radial=A.load()
finally:np.load=old
assert not attempts
source=Path(E.__file__);stub=types.ModuleType('m43');stub.Ops=E.Ops;prior=sys.modules.get('m43');sys.modules['m43']=stub;baseline=types.ModuleType('baseline');baseline.__file__=str(source);np.load=reject
try:exec(compile(source.read_text(),str(source),'exec'),baseline.__dict__)
finally:
 np.load=old
 if prior is None:sys.modules.pop('m43',None)
 else:sys.modules['m43']=prior
checked=[]
for node in ast.parse(source.read_text()).body:
 if isinstance(node,ast.FunctionDef) and node.name!='_cell_error':assert marshal.dumps(getattr(E,node.name).__code__)==marshal.dumps(getattr(baseline,node.name).__code__);checked.append(node.name)
for name in ['__init__','from_normals','tight']:assert marshal.dumps(getattr(E.Cone,name).__code__)==marshal.dumps(getattr(baseline.Cone,name).__code__);checked.append('Cone.'+name)
d=json.loads((P/'controls.json').read_text());assert len(d['models'])==7;returned=0;unresolved=[]
for r in d['models']:
 c=r['capture'];ev=r['evaluation'];N=c['N'];assert c['copied_float64_elements']==16*N and c['ray_storage_bytes']==128*N and ev['N']==N and c['ignored_zero_callback_score']==1
 ledger=Counter()
 for row in ev['rows']:
  ledger.update(row['center']['setup_selection_counts']);ledger.update(row.get('repeated_determinant_counts',{}));ledger.update(row.get('kernel_counts',{}))
 for event in ev['cold_rules']:ledger.update(event['counts'])
 ledger['universal_constant_arithmetic']+=6;assert dict(ledger)==ev['known_mass_counts'] and sum(ledger.values())==ev['known_mass_total']
 if ev['complete']:
  assert len(ev['rows'])==N==216;mass=sum((F(v['value']) for v in ev['rows']),F(0));bound=sum((F(v['bound']) for v in ev['rows']),F(0));assert abs(float(1-mass)-.25)<=float(bound);assert abs(float(1-mass)-ev['accuracy'])<2e-16;assert float(bound)==ev['conditional_bound'];returned+=1
 else:
  assert r['name']=='alloff' and N==72 and len(ev['rows'])==1 and ev['rows'][0]['mode']=='unresolved';unresolved.append(r['name'])
assert returned==6 and unresolved==['alloff']
out=dict(no_array_load_attempts=attempts,unchanged_compiled_functions=len(checked),analytic_models=7,returned=returned,finite_cap_unresolved=unresolved,review_cpu=time.process_time()-start,scope='Import-only bytecode comparison and saved analytic sums/counts; no mass integrations. Initial incidental-weight import preserved, generic total-return remains false, floating/cost scope unchanged.')
Path(__file__).with_name('positive_adapter_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
