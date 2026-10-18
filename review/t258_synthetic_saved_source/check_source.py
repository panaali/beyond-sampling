from pathlib import Path
import ast,hashlib,json
P=Path('/workspace/work/recover-small_h2_78/review/t258_synthetic_saved_78')
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
s=json.loads((P/'scope.json').read_text())
for f,h in s['pins'].items():assert sha(f)==h,f
current=ast.parse((P/'check.py').read_text());old=ast.parse((P/'check_initial_unrun.py').read_text())
functions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef) and n.name!='main'}
a,b=functions(current),functions(old);assert a==b and len(a)==17
files=[P/n for n in ['check.py','scope.json','README.md']]
r={'complete':True,'pass_all':True,'execution_authorized':True,'source':'5f2f0774','pins':{str(f):sha(f) for f in files},'scope':'One saved-only 138-row combined audit; original132/264 and continuation6/14 preserved separately.19internal19soft20hardCPU2GiB25wall after review lock, pinned Python/NumPy runtime, no producer calls. Publish immutable completed inputs/execution gate first.','checks':['17 independent helper ASTs unchanged from initial unrun checker','scalar cell polynomial and feature order','exact dyadic tail and independent integer nearest-even binary32','Fraction readout and first-max choices','ten malformed argument/reason records without calling producer','all direct owner/continuation/result/progress/gate/reviewer pins mandatory','exact prior failure/coverage/progress missing-event contracts','all source pin hashes and failure retention'],'limitations':['Ten shared NumPyquicksort operations do not independently validate library tie ordering','Scalar clock comparisons use fixed4e-15 tolerance','No bank fitting/actual hinge precision or trained-model inference','Original failed run remains failed; progress totals are two separate streams']}
Path(__file__).with_name('source_gate.json').write_text(json.dumps(r,indent=2)+'\n')
print('PASS:17 helper ASTs unchanged; source/pins only, no output decode')
