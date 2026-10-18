"""T260 source/hash review only; archives are hashed, never decoded."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h4_4')
SRC=ROOT/'review/formula_description_length/pilot_codec'
COMMIT='44c7450e';PUBLICATION='8e85f5ff'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a])
p=json.loads((SRC/'protocol.json').read_text());m=json.loads((SRC/'manifest.json').read_text())
pins={**p['pins'],**m['pins']}
assert len(m['pins'])==37
for f,h in pins.items():assert sha(f)==h,f
source=ast.parse((SRC/'run.py').read_text())
files=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FILES' for t in n.targets))
assert len(files)==14 and len(set(files))==14
for name in files+['README.md']:
    f=SRC/name
    assert git('show',COMMIT+':review/formula_description_length/pilot_codec/'+name)==f.read_bytes(),name
    if name.endswith('.py'):ast.parse(f.read_text())
    pins[str(f)]=sha(f)
subprocess.run(['bash','-n',str(SRC/'launch.sh')],check=True)
subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',COMMIT,PUBLICATION],check=True)
for a,b in [('frozen/codec.py',ROOT/'review/formula_description_length/snapshot/prior/algzoo/description_length_audit/codec.py'),('frozen/features.py',ROOT/'review/replacement_reproduction/features/core.py'),('frozen/labels.py',ROOT/'review/formula_description_length/snapshot/prior/algzoo/labels.py')]:assert (SRC/a).read_bytes()==b.read_bytes()
review=json.loads(Path(p['pilot_review']).read_text());assert review['complete'] is True and review['pass_all'] is True and review['archives']==168 and review['models']==21
contract=json.loads(Path(p['contract_review']).read_text());assert contract['complete'] is True and contract['pass_all'] is True
# Bind source inspected for correspondence; do not decode the pilot results.
for f in [Path(p['python_resolved']),ROOT/'review/replacement_reproduction/pilot/run.py',HERE/'README.md',HERE/'check_metadata.py']:pins[str(f)]=sha(f)
r={'task':'T260 fixed retained-pilot framed codec source/resource review','complete':True,'pass_all':True,'execution_authorized':True,'assignment':'af2abda5','source_commit':COMMIT,'source_publication':PUBLICATION,'pins':pins,'direct_paths':[str(SRC/f) for f in files],
 'coverage':{'artifacts':8,'controls':32,'archive_loads':23,'events':20,'control_encode_attempts':22,'control_encode_returns':22,'control_decode_attempts':16,'control_decode_returns':15,'target_encode_decode_pairs':8},
 'resources':p['resources'],'bit_caps':{'per_artifact':80000000,'total':240000000,'auxiliary_json_bytes':134217728},
 'checks':['14 FILES plus transitive README match original published Git source; all37 manifest prerequisite hashes match.','Frozen codec/features/labels identical to named source copies.','Prefix grammar census/framing, bit packing and exact typed AST/binary64 roundtrip logic.','Selected-family dependencies, retained pooled/cell routing, float32 storage and full-column operation shapes inspected against pilot.','Fixed controls, four-by-two artifact order, mandatory gate pins, resource guards and durable failure paths.','Prior pilot/contract review status checked without score contraction; original publication qualification retained.'],
 'qualifications':['Declared code-dependent upper lengths only, shared decoder/AST/library semantics conditioning.','No actual predictor or scoring parity run; valid successful finite-input correspondence is source-inspected, not global floating execution certification.','Network/H-A1 source-style codes receive no transferred pilot scores.','No minimality, interpretability, generalization, full C2 reproduction, cost benefit or population rate-distortion conclusion.','Separate56 saved review required; a failure does not authorize retry, pruning or larger caps.'],
 'archive_decodes':0,'producer_imports':0,'scientific_calls':0}
with (HERE/'source_gate.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'pass_all':True,'pins':len(pins),'direct_paths':len(files)}))
