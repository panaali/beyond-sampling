"""Source provenance only; no scientific payload decode or author imports."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_56')
SOURCE=ROOT/'review/facet_cancellation'
COMMIT='dbdc40d3'
PUBLICATION='598b98c8'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])

p=json.loads((SOURCE/'protocol.json').read_text())
pins=dict(p['pins'])
for name in ('primitive.py','fixtures.py','run.py','PROTOCOL.md','protocol.json','launch.sh'):
    path=SOURCE/name
    raw=path.read_bytes()
    assert raw==git('show',COMMIT+':review/facet_cancellation/'+name)
    pins[str(path)]=digest(path)
    if name.endswith('.py'):
        ast.parse(raw)
for path,value in pins.items():
    assert digest(path)==value,path
for rel,blob in p['inherited_git_blobs'].items():
    assert (ROOT/rel).read_bytes()==git('cat-file','blob',blob),rel
subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',COMMIT,PUBLICATION],check=True)
subprocess.run(['bash','-n',str(SOURCE/'launch.sh')],check=True)
for name in ('README.md','check_metadata.py'):
    pins[str(HERE/name)]=digest(HERE/name)
result={
 'complete':True,'pass_all':True,'execution_allowed':True,
 'task':'T267 fixed oriented-chain controls and H4T4 comparison',
 'assignment':'9a292cb2','saved_reviewer_assignment':'e250f30e',
 'source_commit':COMMIT,'source_publication':PUBLICATION,
 'original_source_commit':'b715c19a','pins':pins,
 'verified_inherited_git_blobs':len(p['inherited_git_blobs']),
 'checks':['signed affine anchor-chain identity and exact facet multiplicities',
 'same-orthant q range and independent degree18 Chebyshev/GL20 real-rule bound',
 'eight moment/control cases and six guard sources, no replay',
 'canonical archived integer and determinant validation repair',
 'durable attempt/result/chain/integral records and no overwrite/retry',
 'component timer qualification and whole-process resource limits',
 'direct source/runtime/dependency SHA and inherited Git-byte bindings'],
 'execution_scope':{'ordered_controls':8,'rejecting_guards':6,'fixed_correct_simplices':5058,
 'gate_simplices_provenance_only':3853,'GL_order':20,'internal_cpu_seconds':100,
 'os_cpu_soft_hard':[110,110],'address_space_bytes':2147483648,'wall_after_primary_lock_seconds':120,
 'threads':1,'attempts':1,'automatic_retry':False},
 'prerequisite':'Publish this review and a pinned immutable producer gate before launch.',
 'qualifications':['Chain identity is not independent original coverage/disjointness/correctness certification.',
 'Ideal truncation bound excludes floating node, weight, normalization, kernel and accumulation errors.',
 'pass_all does not certify floating baseline/reduced agreement or performance success.',
 'Cold component sum and39 named node operations are not full native cost; inherited discovery remains separate.',
 'Separate saved audit assigned to78; no H4T3, broader-anchor or large-model claim.'],
 'scientific_calls':0,'scientific_payload_decodes':0,'author_imports':0
}
with (HERE/'source_gate.json').open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'pass_all':True,'pins':len(pins),'git_blobs':len(p['inherited_git_blobs'])}))
