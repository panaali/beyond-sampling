"""Application source metadata only; scientific archives are hashed, not decoded."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_910')
SRC=ROOT/'review/residual_band_application'
COMMIT='9ec60f45'
PUBLICATION='9ec60f45'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((SRC/'application_protocol.json').read_text())
pins=dict(p['pins'])
old=json.loads(Path(p['controls_review']).read_text())
assert old['complete'] is True and old['pass_all'] is True and old['rows']==18 and old['events']==38
for path,value in old['input_pins'].items():
    assert path not in pins or pins[path]==value
    pins[path]=value
for path,value in pins.items():assert sha(path)==value,path
names=['run_application.py','application_math.py','application_protocol.json','APPLICATION.md','application_launch.sh']
for name in names:
    path=SRC/name
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',COMMIT+':review/residual_band_application/'+name])
    assert raw==path.read_bytes(),name
    if name.endswith('.py'):ast.parse(raw)
    pins[str(path)]=sha(path)
subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',COMMIT,PUBLICATION],check=True)
subprocess.run(['bash','-n',str(SRC/'application_launch.sh')],check=True)
for name in ['README.md','check_metadata.py']:pins[str(HERE/name)]=sha(HERE/name)
r={'complete':True,'pass_all':True,'execution_authorized':True,
 'task':'T278 one fixed H4T4 residual-band numerical discriminator','assignment':'9f66c544',
 'source_commit':COMMIT,'source_publication':PUBLICATION,'original_application_source':'fadd5157',
 'repairs':['dc3dfcdb','a6f5ab04','9ec60f45'],'pins':pins,'direct_paths':names,
 'checks':['coarea density/Jacobian and uniform configuration factors',
 'R23/R12 windows, half-side second-rank labels and delta coordinate',
 'exact actual/frozen intervals, signed bands and positive-weight remainder aggregation',
 'synthetic formula-parity reuse and reference artifact/dimension/hex binding',
 'fixed GL3/5 catalogue with18 reused centres and predetermined64 full-cell records',
 'returned face/partition/error/formula durability, raw caps and final counts',
 'predeclared scheduling-only centre projection and explicit composed-cost omissions'],
 'execution_scope':{'model':'Pinned exact-dyadic H4T4 seed0 only','rules':[3,5],
 'full_success_unique_points':2718,'full_success_partitions':5436,'full_success_events':5439,
 'scheduling_points':18,'projection_stop_above_cpu':75,'internal_cpu':90,'soft_hard_cpu':[105,110],
 'as_soft_hard':[2**31,2**31],'wall_after_lock':120,'lock':'/workspace/work/codex-compute.lock',
 'completed_raw_byte_cap':128*2**20,'attempts':1,'automatic_retry':False},
 'prerequisite':'Publish explicit numerical execution scope and pinned immutable application gate before launch.',
 'qualifications':['Finite interpolated/truncated face law and outer quadrature are not enclosed.',
 'Taylor bounds exclude floating evaluation; direct parity is only an inner diagnostic.',
 'Composed CPU excludes observation/verification/aggregation and is not FPE or full runtime.',
 'Reference is inherited conditional advice; no fresh truth or historical runtime reproduction.',
 'A scheduling stop supports only the completed prefix; independent saved audit remains required.',
 'Only64 predetermined full-cell samples, not exhaustive independent recurrence verification.'],
 'scientific_calls':0,'scientific_archive_decodes':0,'author_imports':0}
with (HERE/'source_gate.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'complete':True,'pass_all':True,'pins':len(pins),'prior_input_pins':len(old['input_pins'])}))
