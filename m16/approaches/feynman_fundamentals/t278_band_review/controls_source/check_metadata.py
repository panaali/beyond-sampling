"""T278 source-only provenance checker; never imports producer modules."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_910')
SRC=ROOT/'review/residual_band_application'
COMMIT='c2f4534e'
PUBLICATION='28cde6b7'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((SRC/'controls_protocol.json').read_text())
pins=dict(p['pins'])
for path,value in pins.items():assert sha(path)==value,path
names=['intervals.py','gaussian.py','controls.py','controls_launch.sh','controls_protocol.json','PLAN.md','CONTROLS.md']
for name in names:
    path=SRC/name
    old=subprocess.check_output(['git','-C',str(ROOT),'show',COMMIT+':review/residual_band_application/'+name])
    assert old==path.read_bytes(),name
    if name.endswith('.py'):ast.parse(old)
    pins[str(path)]=sha(path)
subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',COMMIT,PUBLICATION],check=True)
subprocess.run(['bash','-n',str(SRC/'controls_launch.sh')],check=True)
for name in ['README.md','check_metadata.py']:pins[str(HERE/name)]=sha(HERE/name)
r={'complete':True,'pass_all':True,'execution_authorized':True,
   'task':'T278 fixed primitive controls','assignment':'35a5a2e2',
   'source_commit':COMMIT,'source_publication':PUBLICATION,'pins':pins,
   'direct_paths':names,'counts':p['counts'],
   'scope':'One fixed18-row synthetic controls process only; target application separately gated.',
   'resources':{'internal_cpu':17,'os_cpu_soft_hard':[19,20],'as_soft_hard':[2**31,2**31],'wall_after_review_lock':25,'attempts':1},
   'checks':['exact affine root splitting and fixed-mask continuation',
             'first-index positive-volume ties and null endpoint qualification',
             'disjoint signed difference and negative orientation',
             'delta/standard-normal coordinate conversion and both global Taylor bounds',
             'fixed fixtures, counters, runtime/hash guards and durable outputs'],
   'qualifications':['No outward floating CDF certificate.',
                     'No target face-law, model, outer quadrature or performance clearance.',
                     'Finite guard coverage, not general arbitrary-input validation.',
                     'Independent saved audit and separate target source gate remain required.'],
   'author_imports':0,'scientific_calls':0,'target_payload_decodes':0}
with (HERE/'source_gate.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'pass_all':True,'pins':len(pins),'direct_paths':len(names)}))
