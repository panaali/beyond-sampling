"""Source metadata verification only. Never decode retained scientific JSON."""
import ast, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_78')
SRC=ROOT/'review/retained_rate_audit'
COMMIT='e2eaa91c';PUBLICATION='b475ffa9';BRIDGE='42817b64'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
pins={}
manifest=json.loads((SRC/'manifest.json').read_text())
assert manifest['complete'] is True and len(manifest['records'])==45
assert len({r['local'] for r in manifest['records']})==45
for r in manifest['records']:
    p=SRC/r['local'];raw=p.read_bytes()
    assert len(raw)==r['bytes'] and sha(p)==r['sha256'],str(p)
    assert git('rev-parse',r['revision']+':'+r['source']).decode().strip()==r['blob']
    assert git('cat-file','blob',r['blob'])==raw
    subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',r['revision'],BRIDGE],check=True)
    pins[str(p)]=sha(p)
assert git('diff','--name-only',BRIDGE+'^1',BRIDGE)==b''
assert git('diff','--name-only','b08b9a33^1','b08b9a33')==b''
for n in ('check.py','PROTOCOL.md','launch.sh','manifest.json','runtime.json'):
    p=SRC/n
    assert git('show',COMMIT+':review/retained_rate_audit/'+n)==p.read_bytes(),n
    pins[str(p)]=sha(p)
ast.parse((SRC/'check.py').read_text())
subprocess.run(['bash','-n',str(SRC/'launch.sh')],check=True)
subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',COMMIT,PUBLICATION],check=True)
runtime=json.loads((SRC/'runtime.json').read_text())
assert len(runtime['pins'])==88
assert str(Path(runtime['invoked_python']).resolve())==runtime['resolved_python']
assert runtime['resolved_python'] in runtime['pins']
assert {str(p) for p in Path(runtime['mpmath_root']).rglob('*.py')} <= set(runtime['pins'])
for p,h in runtime['pins'].items():assert sha(p)==h,p;pins[p]=h
for n in ('README.md','check_metadata.py','metadata_preflight.json'):pins[str(HERE/n)]=sha(HERE/n)
r={'complete':True,'pass_all':True,'execution_authorized':True,'task':'T283 fixed retained scalar rate audit','assignment':'1abaf48d','source_commit':COMMIT,'source_publication':PUBLICATION,'initial_source':'42653a4b','pins':pins,'direct_paths':[str(SRC/n) for n in ('check.py','PROTOCOL.md','launch.sh','manifest.json','runtime.json')],
 'metadata_checks':{'git_blobs':45,'runtime_hashes':88,'zero_tree_history_bridge':BRIDGE,'source_ast':True,'launcher_syntax':True},
 'coverage':{'t271_rows':289,'t271_groups':22,'t277_rows':720,'t277_cells':90,'t277_models':18},
 'resources':{'internal_cpu':17,'os_cpu':[19,20],'as_bytes':[2147483648,2147483648],'wall_seconds':25,'lock':'/workspace/work/codex-review.lock'},
 'source_checks':['Exact binary64 moment contractions and independent OLS/Student inversion match frozen definitions.','Fixed row, model, budget and seed coverage plus source reference choices.','Full three-seed pooled coverage and preregistered finite rule; retain positivity exclusions.','Raw/computed records precede comparison assertions; atomic failure evidence and attempt-ownership guard.','Source-review hash, direct prerequisite membership and runtime verification before numerical import.'],
 'qualifications':['Source review only: no retained numerical row, fit or prefactor was recomputed.','Fixed saved-display tolerances are consistency allowances, not error enclosures.','Residual regression CIs across budgets do not prove independent-scramble uncertainty or asymptotic rate/equivalence.','Shared reference/scramble effects and budget-dependent allocations remain; CI inclusion of one half is not equivalence.','Prefactor numeric matching is not absent-generator or key-semantics reconstruction; T277 enumeration aggregate remains provenance only.','D6 excluded; native cost and model/runtime historical provenance remain unverified.','One attempt only after published immutable gate; independent saved review remains separate.'],
 'producer_imports':0,'scientific_payload_decodes':0,'scientific_calls':0}
with (HERE/'source_gate.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'pass_all':True,'pins':len(pins)}))
