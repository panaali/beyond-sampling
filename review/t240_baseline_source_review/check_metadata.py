"""Hash/AST/metadata audit only: no archive decoding or author imports."""
from pathlib import Path
import ast, hashlib, json, time
B=Path('/workspace/work/recover-small_h2_78/review/easy_correct_certificate/saved_baseline')
O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
t=time.process_time()
p=read(B/'protocol.json')
for path,h in p['pins'].items():assert sha(path)==h,path
assert len(p['pins'])==96
ast.parse((B/'run.py').read_text())
r=read(p['prior_review']);s=read(p['prior_review_scope'])
assert r['complete'] and r['rows_verified']==41472 and r['independent_affine_recurrences']==128
assert r['run_scope_sha256']==sha(p['prior_review_scope'])
root=Path(p['source'])
required={str(root/x) for x in ['application/results.json','application_protocol.json','application_gate.json','controls_results.json']}
required|={str(root/'application'/f'H8T4seed0_{k:03}.json.gz') for k in range(81)}
assert len(required)==85
for path in required:assert s['pins'][path]==p['pins'][path]
toy=read(p['synthetic_review']);assert toy['complete'] and toy['pass_all'] and len(toy['rows'])==31
assert not (B/'output').exists()
pins={str(B/f):sha(B/f) for f in ['run.py','launch.sh','PROTOCOL.md','protocol.json']}
pins.update(p['pins'])
result=dict(complete=True,pass_all=True,execution_authorized=True,status='passed',task='T240 fixed H8 saved-baseline source/resource review',author_source='7d874a2f',source_publication='9462e9f9',pins=pins,metadata_checks=dict(protocol_pins=96,inherited_review_bindings=85,synthetic_rows=31,archive_decodes=0,author_imports=0,model_calls=0),scope=dict(boxes=20736,archives=81,gaps_quarters=[0,1,2,4],one_run=True,internal_cpu=17,soft_cpu=19,hard_cpu=20,address_space_bytes=2147483648,post_lock_wall_seconds=25),review_findings=['First-index margin signs match correct strict/non-strict ownership.','For positive gap, inner tests imply one unique ordered pair; outer interval feasibility conservatively includes boundary intersections.','At gap zero use the inherited exchangeable second-bin group and denominator divisible by all group sizes, not the positive-gap inner shortcut.','Selected correct/wrong events use inner boxes only; boundary boxes are unresolved.','Exact lower/upper event and unresolved intervals plus conditional accuracy [cL/pU,1-wL/pU] are conservative under inherited valid mass and margin premises.','Protocol/gate pins verified separately; prior independent review binds original result and all selected archive bytes.','Durable chunk and failure records, no overwrite, fixed resource launcher and no expansion.'],limitations=['No archive decode, original enclosure recomputation, mass integration, recurrence, model, RNG or performance measurement by reviewer.','Inherited margin review recomputed 128 recurrences, not all margins independently.','No partial-box intersection masses; conditional certificate utility remains unknown pending the sole saved run and independent audit.','Source-work categories omit repeated hash I/O and individual accumulator costs; no complete cost or target H16 claim.'],cpu_seconds=time.process_time()-t)
(O/'source_gate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=True,pins=len(pins),metadata_cpu=result['cpu_seconds'])))
