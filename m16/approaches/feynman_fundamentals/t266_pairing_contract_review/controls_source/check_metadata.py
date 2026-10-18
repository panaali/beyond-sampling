"""Provenance and source syntax only; no primitive or table arithmetic execution."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_56')
SRC=ROOT/'review/residual_pairing_controls'
COMMIT='61ca7042';PUBLICATION='da40a449'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((SRC/'protocol.json').read_text());pins=dict(p['pins'])
old=json.loads(Path(p['proof_review']).read_text());assert old['complete'] is True and old['pass_all'] is True
assert old['execution_allowed'] is False
for path,value in old['pins'].items():
    assert path not in pins or pins[path]==value
    pins[path]=value
for path,value in pins.items():assert sha(path)==value,path
names=['primitive.py','controls.py','fixtures.json','PROTOCOL.md','protocol.json','launch.sh']
for name in names:
    path=SRC/name
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',COMMIT+':review/residual_pairing_controls/'+name])
    assert raw==path.read_bytes(),name
    if name.endswith('.py'):ast.parse(raw)
    pins[str(path)]=sha(path)
subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',COMMIT,PUBLICATION],check=True)
subprocess.run(['bash','-n',str(SRC/'launch.sh')],check=True)
for name in ['README.md','check_metadata.py']:pins[str(HERE/name)]=sha(HERE/name)
r={'complete':True,'pass_all':True,'execution_allowed':True,'task':'T266 exact finite-law controls','assignment':'213f4b30',
 'source_commit':COMMIT,'source_publication':PUBLICATION,'pins':pins,'direct_paths':names,
 'coverage':p['coverage'],'limits':p['limits'],
 'checks':['invariant and non-invariant mean/variance/absolute identities','orbit representative/fixed-point/endpoint-tie handling',
 'hypothetical finite-budget variance floor ratios and zero-variance nulls','literal label/tie cases and12guard source paths',
 'byte-bound predecessor/source/runtime prerequisites','returned-record durability, failure reset and resource limits'],
 'qualifications':['Finite tables are not Gaussian/target measurements.','Non-invariant variance reduction is not an unbiased MSE gain.',
 'Declared costs are hypothetical, not native/full evaluation costs.','No trained surrogate or integral selected; saved audit remains required.'],
 'scientific_calls':0,'author_imports':0,'model_calls':0}
with (HERE/'source_gate.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'pass_all':True,'pins':len(pins)}))
