"""T268 source provenance only; no saved scientific row decoding."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_910')
SRC=ROOT/'review/blind_sector_accuracy'
COMMIT='26c93a5e';PUBLICATION='dd2f330a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((SRC/'protocol.json').read_text());pins=dict(p['pins'])
for path,value in pins.items():assert sha(path)==value,path
names=['sector.py','fixtures.py','run.py','launch.sh','PROTOCOL.md','protocol.json']
for name in names:
    path=SRC/name
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',COMMIT+':review/blind_sector_accuracy/'+name])
    assert raw==path.read_bytes(),name
    if name.endswith('.py'):ast.parse(raw)
    pins[str(path)]=sha(path)
subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',COMMIT,PUBLICATION],check=True)
subprocess.run(['bash','-n',str(SRC/'launch.sh')],check=True)
pins[p['python_resolved']]=sha(p['python_resolved'])
for name in ['README.md','check_metadata.py']:pins[str(HERE/name)]=sha(HERE/name)
r={'complete':True,'pass_all':True,'execution_authorized':True,'task':'T268 fixed sign-sector saved contraction','assignment':'57f443c6',
 'source_commit':COMMIT,'source_publication':PUBLICATION,'original_source':'f875b6eb','numerical_repair':'f9379258',
 'pins':pins,'direct_paths':names,'counts':p['counts'],'resources':p['resources'],
 'checks':['exact orthant assignment including zero boundary coordinates','normalized determinant and saved interval sequence binding',
 'physical interval intersections and exact concave binary-loss extrema','17control sources and fixed5058/all16/five-sector coverage',
 'computed-evidence durability and hexadecimal publication guards','explicit predicted-class confound and inherited coverage qualifications'],
 'qualifications':['Only binary correctness-indicator sufficiency; true labels vary within sectors, so predicted-class laws remain unresolved.',
 'Original exact correctness coverage/disjointness and outward numerical/platform assumptions remain inherited.',
 'Historical reference cost is advice, not a cheap estimator or native-cost saving.',
 'No Gaussian/model/geometry replay, no T10/HP1 transfer, separate h4 saved audit required.'],
 'scientific_calls':0,'scientific_payload_decodes':0,'producer_imports':0}
with (HERE/'source_gate.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'pass_all':True,'pins':len(pins)}))
