"""Reconstruct expected fake-provider reporting records without author imports."""
import hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=Path('/workspace/work/recover-small_h2_910/review/evalH_stats_review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.process_time();p=json.loads((D/'protocol.json').read_text());out=json.loads((D/'controls.json').read_text());gate=json.loads((HERE/'evalH_stats_source_gate.json').read_text())
 assert p['source_gate'] not in (None,'pending','')
 for path,h in p['pins'].items():assert sha(Path(path))==h,path
 assert out['protocol_sha256']==sha(D/'protocol.json') and out['source_sha256']==sha(D/'controls.py')
 assert out['complete'] is True and out['active'] is None and 'failure' not in out
 assert out['new_real_model_estimator_mass_MC_calls']==0 and len(out['rows'])==6
 assert p['cases']==gate['cases']
 totals={'row_pairs':0,'fake_estimator_calls':0,'fake_truth_calls':0,'fake_rescale_calls':0};details=[]
 for c,rec in zip(p['cases'],out['rows'],strict=True):
  assert rec['case']==c and rec['all_checks_pass'] is True
  N=c['N'];events=[];models=[];rows={'old':[],'fixed':[]};selected=0;changed=0
  for i in range(N):
   stats={'cells':0 if c['zero'] else i+1,'cut':0 if c['zero'] else 2*i+3,'resolved':0 if c['zero'] else i}
   models.append({'i':i,'rescaled':False,'stats':stats,'cost':200+i,'value':.5+i/128})
   events.extend([['truth',i],['est',i,False]])
   if i%8==0:events.extend([['rescale',i],['est',i,True]]);selected+=1
   fixed={'truth':.25+i/128,'est':.5+i/128,'ops':200+i,'exact_ops':100+i,'leaves':50+i,**stats}
   old=dict(fixed)
   for k,v in stats.items():old[k]+= (0 if c['zero'] else 2*v+3) if i%8==0 else 0
   changed+=old!=fixed;rows['old'].append(old);rows['fixed'].append(fixed)
  final={k:rows['old'][-1][k] for k in ('cut','resolved','cells')}
  for arm in ('old','fixed'):
   expected={'rows':rows[arm],'inv':1/1024 if c['changed_output'] else 0.,'events':events,'final_stats':final,'inputs':models}
   assert rec[arm]==expected,(c['name'],arm)
  assert rec['rescale_calls']==selected
  totals['row_pairs']+=N;totals['fake_estimator_calls']+=2*(N+selected);totals['fake_truth_calls']+=2*N;totals['fake_rescale_calls']+=2*selected
  details.append({'case':c['name'],'rows':N,'rescale_calls_per_arm':selected,'changed_stat_rows':changed,'all_records_exact':True})
 assert totals=={'row_pairs':53,'fake_estimator_calls':128,'fake_truth_calls':106,'fake_rescale_calls':22}
 result={'complete':True,'independent_expected_records':True,'controls_sha256':sha(D/'controls.json'),'protocol_sha256':sha(D/'protocol.json'),'checker_sha256':sha(Path(__file__)),'source_gate_sha256':sha(HERE/'evalH_stats_source_gate.json'),'details':details,'totals':totals,'author_cpu_seconds':out['cpu_seconds'],'review_cpu_seconds':time.process_time()-start,'author_or_estimator_calls_in_review':0,'conclusion':'All six saved synthetic reporting controls match independently reconstructed records exactly. Snapshot isolates original per-model statistics and preserves calls/values/ops/invariance.','limitations':['No historical full-table statistics recovered by these controls.','No model replay, native-cost claim, corrected scaling law, or shared-source integration implied.','Historical clean297 excludes blanket halving; noisy correction requires actual selected-rescale counts.']}
 (HERE/'evalH_stats_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':True,'totals':totals,'review_cpu_seconds':result['review_cpu_seconds']}))
if __name__=='__main__':main()
