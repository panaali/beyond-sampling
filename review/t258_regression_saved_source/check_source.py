from pathlib import Path
import ast,hashlib,json
P=Path('/workspace/work/recover-small_h2_78/review/t258_regression_saved_78')
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
s=json.loads((P/'scope.json').read_text())
for f,h in s['pins'].items():assert sha(f)==h,f
ast.parse((P/'check.py').read_text())
files=[P/n for n in ['check.py','scope.json','README.md']]
r={'complete':True,'pass_all':True,'execution_authorized':True,'source':'6f9b9e14','pins':{str(f):sha(f) for f in files},'scope':'One saved14row30event audit,19internal19soft20hardCPU2GiB25wall after reviewlock. No producer/model/RNG calls; exact independent elimination is reviewer arithmetic. Publish immutable completedinputs and execution gate first.','checks':['literal six fixture and eight guard contracts','independent rational elimination rather than author Cramer determinants','all14 ideal matrix/RHS/coefficient and15 started14completed solve records','source binary regularizer matrix reconstruction','exact normalized residual/error metrics and first/last/unseen routing','codec coefficient bits and saved prediction consistency','conservative8u dot bound for finite tiny non-underflow fixtures','mandatory source/raw/review/progress/deviation pins','producer protocol failure retained irrespective of numerical pass'],'qualification':'Published-gate-before-run condition was not met by author. This review authorizes saved arithmetic only, never retroactive launch clearance.'}
Path(__file__).with_name('source_gate.json').write_text(json.dumps(r,indent=2)+'\n')
print('PASS source/pins only; no raw numeric records decoded')
