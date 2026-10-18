"""Source metadata only; no candidate import or numerical suite execution."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_78')
SRC=ROOT/'review/margin_loss_controls_78'
COMMIT='22b2f8e4';PUBLICATION='c6812df6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((SRC/'protocol.json').read_text())
r=json.loads(Path(p['prior_written_review']).read_text())
assert r['complete'] is True and r['pass_all'] is True
pins={**p['pins'],**r['pins']}
for f,h in pins.items():assert sha(f)==h,f
names=['algebra.py','fixtures.py','run.py','launch.sh','PROTOCOL.md','protocol.json']
for n in names+['README.md']:
    f=SRC/n
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',COMMIT+':review/margin_loss_controls_78/'+n])
    assert raw==f.read_bytes(),n
    if n.endswith('.py'):ast.parse(raw)
    pins[str(f)]=sha(f)
for n in ('algebra.py','fixtures.py','run.py','launch.sh'):
    before=subprocess.check_output(['git','-C',str(ROOT),'show','06c7cf14:review/margin_loss_controls_78/'+n])
    assert before==(SRC/n).read_bytes(),n
subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',COMMIT,PUBLICATION],check=True)
subprocess.run(['bash','-n',str(SRC/'launch.sh')],check=True)
assert str(Path(p['invoked_python']).resolve())==p['resolved_python']
assert sha(p['resolved_python'])==p['python_sha256'];pins[p['resolved_python']]=p['python_sha256']
for n in ('README.md','check_metadata.py'):pins[str(HERE/n)]=sha(HERE/n)
out={'task':'T259 exact synthetic margin-loss controls source/resource gate','assignment':'7459ae50','complete':True,'pass_all':True,'execution_authorized':True,'source_commit':COMMIT,'source_publication':PUBLICATION,'initial_source':'06c7cf14','pins':pins,'direct_paths':[str(SRC/n) for n in names],
 'coverage':{'cases':17,'solve_attempts':36,'solve_returns':35,'expected_singular_failures':1,'events':36,'max_square_system_dimension':4},
 'resources':{'internal_cpu':15,'CPU':[16,20],'AS':[2147483648,2147483648],'wall':25,'lock':'/workspace/work/codex-review.lock'},
 'checks':['Exact normal equations and Gauss-Jordan pivot/failure recording.','Independent pair-design zero-sum gauge, all four pair conventions and two sample normalizations, matched penalty blocks.','Manual analytic fixture coefficients, predictions, first-index ties and four separating exceptions.','36/35 solve ledger and36 event coverage, no extra malformed-domain campaign.','Source/prior/reviewer/runtime binding before candidate imports, no-overwrite ownership and durable partial output.'],
 'qualifications':['Clears only one fixed17-case exact synthetic run after published immutable execution gate.','No trained/model/array/RNG/Gaussian or HP1/HP2 calls, no99percent or feature-ceiling conclusion.','Prior exact theorem remains restricted to common fixed feature/routing/penalty/linear-solution maps.','General floating parity and alternative weighted/nonlinear objectives remain untested.','56 independent saved arithmetic review required; failures do not authorize automatic retries.'],
 'producer_imports':0,'candidate_calls':0,'scientific_payload_decodes':0}
with (HERE/'source_gate.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'pass_all':True,'pins':len(pins),'direct_paths':len(names)}))
