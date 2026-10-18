"""Only source metadata and saved-prerequisite status; no scientific archive decode."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_56')
SRC=ROOT/'review/facet_cancellation/h4t3'
commit,publication=sys.argv[1:]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((SRC/'protocol.json').read_text())
pins=dict(p['pins'])
old=json.loads(Path(p['prior_saved_review']).read_text())
assert old['complete'] is True and old['pass_all'] is True
assert len(old['controls'])==8 and old['guards']==6
assert all(x['passed'] is True for x in old['controls'])
for path,value in old['input_pins'].items():
    assert path not in pins or pins[path]==value
    pins[path]=value
for path,value in pins.items():assert sha(path)==value,path
names=['angle.py','run.py','PROTOCOL.md','protocol.json','launch.sh']
for name in names:
    path=SRC/name
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':review/facet_cancellation/h4t3/'+name])
    assert raw==path.read_bytes(),name
    if name.endswith('.py'):ast.parse(raw)
    pins[str(path)]=sha(path)
subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',commit,publication],check=True)
subprocess.run(['bash','-n',str(SRC/'launch.sh')],check=True)
for name in ['README.md','check_metadata.py']:pins[str(HERE/name)]=sha(HERE/name)
r={'complete':True,'pass_all':True,'execution_allowed':True,'task':'T267 fixed H4T3 second rung',
 'assignment':'6c62a50c','source_commit':commit,'source_publication':publication,'original_source':'1d8b035d',
 'pins':pins,'direct_paths':names,'prior_controls_reused':8,'prior_guards_reused':6,
 'checks':['positive trihedral atan2 branch and exact determinant orientation','six fixed analytic angle fixture sources',
 'unchanged exact H4T3 capture paths, identifiers and tie semantics','dtype width prevents silent binary64 narrowing',
 'prior saved control status and every inherited input hash','fresh-output/durable-capture/partial-failure paths',
 'actual runtime limits, component/full-cost qualifications and single bounded launch'],
 'execution_scope':{'fresh_angle_controls':6,'new_geometry_captures':1,'H':4,'T':3,'internal_cpu':17,
 'geometry_cpu':12,'soft_hard_cpu':[20,20],'as_soft_hard':[2**31,2**31],'wall_after_review_lock':25,'attempts':1},
 'qualifications':['Containment/extremality and determinant sums are not independent completeness/disjointness proof.',
 'No outward floating solid-angle certificate or matched full-native speed claim.',
 'Archive bytes authoritative; historical checkpoint/runtime identity remains inherited.',
 'Independent saved audit required; no automatic retry or broader-model inference.'],
 'scientific_calls':0,'weight_payload_decodes':0,'author_imports':0}
with (HERE/'source_gate.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'pass_all':True,'pins':len(pins),'inherited_input_pins':len(old['input_pins'])}))
