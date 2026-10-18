"""Independent saved T221 lifecycle/interface audit; no estimator or RNG calls."""
from pathlib import Path
import ast,hashlib,json,time,traceback
HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h2_56/review/srb_draw_diagnostics')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.process_time();dest=HERE/'t221_controls_saved_review.json';assert not dest.exists()
 r=read(BASE/'controls/results.json');p=read(BASE/'protocol.json');g=read(BASE/'execution_gate.json')
 assert r['complete'] and 'failure' not in r and r['gate']==g and r['protocol_sha256']==sha(BASE/'protocol.json')
 assert g['clearance'] and g['protocol_sha256']==sha(BASE/'protocol.json') and r['runtime']==p['runtime']
 for pins in [p['pins'],g['pins'],read(BASE/'source_manifest.json')['pins'],read(p['prior_protocol'])['pins']]:
  for path,digest in pins.items():assert sha(path)==digest,path
 sr=read(g['source_review']['path']);assert sr['status']=='passed' and sr['source_review_complete']
 old=read(p['prior_results']);priorreview=read(p['prior_saved_review']);assert old['complete'] and priorreview['complete']
 assert priorreview['pins'][p['prior_saved_review_result_key']]==sha(p['prior_results'])
 oldprotocol=read(p['prior_protocol']);source=ast.parse(Path(oldprotocol['cv_source']).read_text())
 selected=[]
 for n in source.body:
  if isinstance(n,ast.FunctionDef) and n.name in ['pstratum','pc','draw_all','make']:selected.append(n)
  elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['T','EDGES','NS','_pc','COST_DRAW'] for t in n.targets):selected.append(n)
 tree=ast.dump(ast.Module(body=selected,type_ignores=[]),include_attributes=False)
 assert hashlib.sha256(tree.encode()).hexdigest()==p['numeric_ast_sha256']==r['numeric_ast_sha256']
 assert ast.dump(ast.parse(read(p['prior_trees'])['composition']),include_attributes=False)==tree
 adapter=ast.parse((BASE/'adapter.py').read_text());snapshot=next(n for n in adapter.body if isinstance(n,ast.ClassDef) and n.name=='Snapshot')
 assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dataclass' and any(k.arg=='frozen' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords) for n in snapshot.decorator_list)
 cases=read(p['fixtures'])['cases'];assert len(cases)==len(old['rows'])==len(r['rows'])==4
 baseline={x['name']:x for x in old['rows']};all_snapshots=[];expected_events=[dict(phase='attempt',stage='preflight'),dict(phase='result',stage='preflight')]
 def snapshot_check(s,status,quota,requests,drawn,kept,cost,error):
  assert s==dict(status=status,quota=quota,requests=requests,requested_candidates=sum(x[0][0] for x in requests),completed_candidates=sum(x[1][0] for x in requests if x[1] is not None),source_drawn=drawn,kept=kept,reported_cost=cost,error_type=error)
  all_snapshots.append(s)
 for c,oldrow,row in zip(cases,old['rows'],r['rows'],strict=True):
  assert c['name']==oldrow['name']==row['name'];prior=oldrow['arms'][0]
  for k in ['estimate_hex','cost','requests','warnings']:assert row[k]==prior[k]
  assert row['parity'] and row['diagnostics_match'] and row['cache_identity_preserved']
  requests=[[x['shape'],x['shape'],None] for x in prior['requests']]
  assert all(x[0]==[50000,9] for x in requests)
  snapshot_check(row['snapshot'],'complete',oldrow['nh'],requests,oldrow['drawn'],oldrow['kept'],oldrow['raw_cost'],None)
  assert row['cost']==oldrow['raw_cost']
  expected_events += [dict(phase='attempt',name=c['name']),dict(phase='result',name=c['name'],parity=True,diagnostics_match=True)]
 exceptions=r['exception_controls'];assert len(exceptions)==3
 quota=baseline['one_batch_1000']['nh']
 for j,name in enumerate(['rng_before_return','rng_after_one_batch','Q_after_draw']):
  row=exceptions[j];assert row['name']==name and row['caught']=='SentinelError' and row['passed']
  if j==0:
   requests=[[[50000,9],None,'SentinelError']];completed=[];drawn=kept=None
  elif j==1:
   requests=[[[50000,9],[50000,9],None],[[50000,9],None,'SentinelError']];completed=baseline['two_batches_1000']['arms'][0]['requests'][:1];drawn=kept=None
  else:
   requests=[[[50000,9],[50000,9],None]];completed=baseline['one_batch_1000']['arms'][0]['requests'];drawn=50000;kept=584
  assert row['completed_requests']==completed and row['synthetic_Q_calls']==int(j==2)
  snapshot_check(row['snapshot'],'failure',quota,requests,drawn,kept,None,'SentinelError')
  expected_events += [dict(phase='attempt',name=name),dict(phase='result',name=name,passed=True)]
 row=r['nested_control'];minimum=baseline['minimum_quota'];assert row['nested_errors']==['nested observation unsupported'] and row['immutability_exception']=='FrozenInstanceError'
 assert row['estimate_hex']==minimum['arms'][0]['estimate_hex'] and row['cost']==minimum['raw_cost'] and row['requests']==minimum['arms'][0]['requests']
 snapshot_check(row['snapshot'],'complete',minimum['nh'],[[[50000,9],[50000,9],None]],50000,28,minimum['raw_cost'],None)
 expected_events += [dict(phase='attempt',name='nested_and_immutable'),dict(phase='result',name='nested_and_immutable',passed=True),dict(phase='complete')]
 events=list(map(json.loads,(BASE/'controls/progress.jsonl').read_text().splitlines()));assert events==expected_events
 assert r['original_Q_estimator_calls']==4 and r['synthetic_Q_calls']==2 and r['framework_forward_calls']==r['Gaussian_RNG_calls']==0
 result=dict(complete=True,pass_all=True,task='T221 independent saved interface/lifecycle controls',pins={str(BASE/n):sha(BASE/n) for n in ['controls/results.json','controls/progress.jsonl','protocol.json','execution_gate.json','source_manifest.json','adapter.py','controls.py']},checker_sha256=sha(__file__),success_baseline_rows=4,synthetic_failure_rows=3,nested_lifecycle_rows=1,snapshots=8,original_Q_calls=4,synthetic_Q_calls=2,progress_events=len(events),requested_candidates=sum(x['requested_candidates'] for x in all_snapshots),completed_candidates=sum(x['completed_candidates'] for x in all_snapshots),requested_shapes=sum(len(x['requests']) for x in all_snapshots),completed_shapes=sum(x[1] is not None for s in all_snapshots for x in s['requests']),numeric_ast_sha256=r['numeric_ast_sha256'],review_cpu_seconds=time.process_time()-start,prior_control_cpu_seconds=r['cpu_seconds'],new_author_estimator_rng_calls=0,scope='Saved original baseline bits/cost/requests/warnings reused once, exact request/return/failure accounting and nested rejection/immutable-snapshot records; original numerical AST unchanged.',limitations=['Cache object identity and historical object immutability are runtime assertions recorded by the gated harness, not reconstructible object identities from JSON. Frozen dataclass and per-call copies checked in source.','Failed RNG internals and partial arrays before return are unknown; only requested shapes/completed returns are counted.','Zero-readout original outputs are interface parity, not Q accuracy evidence.','No Gaussian sampling-law, trained-model, runtime-speed or full-cost/multiplier claim.'])
 dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,sha256=sha(dest),cpu=result['review_cpu_seconds'],events=len(events))))
if __name__=='__main__':
 try:main()
 except BaseException:
  (HERE/'t221_saved_reviewer_failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2)+'\n');raise
