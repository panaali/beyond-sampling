"""Saved batch-versus-original primitive records; no primitive execution."""
from pathlib import Path
from collections import Counter
import json,hashlib,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/positive_inner5_batch');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((P/'protocol.json').read_text())
for p,h in protocol['pins'].items():assert sha(Path(p))==h
old=json.loads(Path(protocol['old_results']).read_text());d=json.loads((P/'controls.json').read_text())
assert old['complete'] and d['complete'] and d['all_pass'] and d['input_bytes_unchanged'] and len(d['rows'])==len(old['rows'])==26
supported=0;statuses=Counter()
for prior,r in zip(old['rows'],d['rows'],strict=True):
 assert prior['id']==r['id'] and prior['candidate']==r['expected'] and r['passed'] and r['status_branch_pass']
 a=r['actual'];e=prior['candidate'];assert (a['status'],a['branch'])==(e['status'],e['branch']);statuses[a['status']]+=1
 if e['status']=='ok':assert float(a['value']).hex()==float(e['value']).hex() and r['supported_bits_pass'];supported+=1
 if 'exception' in e:assert a['exception']==e['exception']
assert len(d['guards'])==4 and [r['id'] for r in d['guards']]==['empty','width4','different_N','float32'] and all(r['passed'] and r['error'] for r in d['guards'])
progress=[json.loads(x) for x in (P/'progress.jsonl').read_text().splitlines()];assert len(progress)==35 and progress[1:27]==d['rows']
assert progress[0]['ids']==[r['id'] for r in old['rows']]
for i,r in enumerate(d['guards']):assert progress[28+2*i]==r
assert supported==16 and d['old_scalar_reference_mass_model_calls']==0 and d['new_batch_primitive_calls']==1 and d['new_invalid_guard_calls']==4
out={'pass':True,'cases':30,'supported_bitwise':supported,'statuses':dict(statuses),'author_cpu':d['cpu_seconds'],'candidate_cpu_measured':d['candidate_cpu_seconds'],'review_cpu':time.process_time()-start,'files':{str(P/f):sha(P/f) for f in ('controls.json','progress.jsonl','primitive.py','controls.py','protocol.json')},'scope':'Saved status/branch/exception/value parity and coverage only. No primitive or outer evaluation. Zero timer reading is not zero cost; no speed/native-cost/full-FP claim.'}
(O/'positive_inner_batch_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
