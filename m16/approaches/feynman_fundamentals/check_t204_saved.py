"""Independent archived rows, replacement history and rational budget audit."""
from pathlib import Path
import json,hashlib,ast,time
from fractions import Fraction as F
from collections import Counter
P=Path('/workspace/work/recover-small_h2_910/review/leaderboard_v3_repair')
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda n:json.loads((P/n).read_text())
def main():
 t=time.process_time();d=read('results.json');q=read('protocol.json');g=read('gate.json');m=read('manifest.json')
 assert d['status']=='complete' and d['source_sha']==q['audit_sha256']==sha(P/'audit.py')
 assert sha(P/'manifest.json')==q['manifest_sha256'] and sha(Path(g['artifact']))==g['sha256'] and g['approved']
 for r in m['files']:
  b=(P/r['snapshot']).read_bytes();assert len(b)==r['bytes'] and sha(P/r['snapshot'])==r['sha256']
  assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['git_blob']
 classes={};selected={};dupcounts={}
 for era in ['current','pre_class']:
  tree=ast.parse((P/f'frozen/{era}/m16/leaderboard.py').read_text())
  classes[era]=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CLASS' for x in n.targets))
  latest={};replacements=[]
  for line in (P/f'frozen/{era}/results/leaderboard_v3_raw.jsonl').read_text().splitlines():
   r=json.loads(line)
   if (r['v'],r['H'],r['T'])!=(3,16,10):continue
   key=tuple(r[k] for k in ['name','model','budget','rep'])
   if key in latest:
    digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    replacements.append(dict(key=list(key),identical=latest[key]==r,previous_sha=digest(latest[key]),replacement_sha=digest(r)))
   latest[key]=r
  assert replacements==d['raw_duplicate_inventory'][era]
  selected[era]=latest;dupcounts[era]=len(replacements)
 inventory=[]
 for name in sorted({k[0] for k in selected['current']}):
  rr={k:v for k,v in selected['current'].items() if k[0]==name}
  # Frozen archive contains only legacy schema, checked rather than inferred.
  assert all('det_tested' not in r and type(r['det_ok']) is bool for r in rr.values())
  inventory.append(dict(name=name,current_class=classes['current'].get(name,['unreviewed'])[0],pre_class=classes['pre_class'].get(name,['unreviewed'])[0],rows=len(rr),states=dict(Counter('legacy_recorded_pass' if r['det_ok'] else 'legacy_unknown' for r in rr.values())),identical_pre_class_records=sum(selected['pre_class'].get(k)==r for k,r in rr.items())))
 assert inventory==d['determinism_inventory']
 original=read('frozen/current/results/leaderboard_v3_H16_T10.json');out=read('all_budgets.json');assert len(original)==len(out)==498
 assert len({(r['name'],r['group'],r['budget']) for r in original})==498
 for a,b in zip(original,out,strict=True):
  assert set(b)==set(a)|{'dri_class_at_source_snapshot','determinism_evidence'}
  assert all(b[k]==v for k,v in a.items()) and b['dri_class_at_source_snapshot']==classes['current'].get(a['name'],['unreviewed'])[0]
 lines=[s for s in (P/'ALL_BUDGETS.md').read_text().splitlines() if s.startswith('| ')][1:];assert len(lines)==498
 for line,r in zip(lines,sorted(out,key=lambda x:(x['name'],x['group'],x['budget'])),strict=True):
  fields=[s.strip() for s in line.split('|')[1:-1]]
  assert fields==[str(r['name']),str(r['group']),str(r['budget']),str(r['cls']),str(r['dri_class_at_source_snapshot']),format(r['ratio'],'.12g'),f"[{r['ci'][0]:.12g}, {r['ci'][1]:.12g}]",str(r['tuned'])]
 assert d['report']==dict(rows=498,unique_keys=498,all_original_fields_preserved=True,negative_ratios=sum(r['ratio']<0 for r in original),degenerate_intervals=sum(r['ci'][0]==r['ci'][1] for r in original))
 expected=[({'det_ok':False},'legacy_unknown'),({'det_ok':True},'legacy_recorded_pass'),({'det_tested':False,'det_ok':False},'not_tested'),({'det_tested':True,'det_ok':False},'tested_fail'),({'det_tested':True,'det_ok':True},'tested_pass')]
 assert d['controls']['determinism']==[dict(input=r,expected=s,actual=s) for r,s in expected]
 assert d['controls']['cost_covariance']==dict(own_a=str(F(1,32)/(F(1,4)+F(1,16))),own_b=str(F(1,32)/(F(1,12)+F(3,16))),uniform=str(F(1,32)/(F(1,8)+F(3,32))))
 e44=read('frozen/current/m16/approaches/ladder_training_axis/rung_k24.json')['seeds']['4']['panel'];rr=read('frozen/current/m16/approaches/training_axis_review/absolute_k24_s4.json');s=d['e44_t114']
 assert len(e44['truths'])==len(e44['t75_errs'])==1024 and len(rr['rows'])==256
 numerator=sum(F(x)**2 for x in e44['t75_errs']);variance=sum(F(x)*(1-F(x)) for x in e44['truths']);r44=numerator*F(e44['t75']['ops'])/(72*variance)
 num=F(0);den=F(0);var=F(0);cost=F(0)
 for r in rr['rows']:
  p=F(r['truth']);e=F(r['t75']['value'])-p;c=F(r['t75']['own_ops'],r['t75']['sample_ops']-3);assert 0<p<1 and c>0
  num+=e*e;den+=p*(1-p)/c;var+=p*(1-p);cost+=c
 own=num/den;uniform=num*cost/(256*var)
 assert F(s['e44_uniform_ratio'])==r44 and F(s['t114_own_ratio'])==own
 assert s['e44_ratio']==float(r44) and s['t114_ratio']==float(own) and s['t114_uniform_ratio']==float(uniform)
 source_summary=next(r for r in read('frozen/current/m16/approaches/training_axis_review/summary.json')['cells'] if r['file']=='absolute_k24_s4.json')
 assert s['e44_saved_ci']==e44['t75']['ci'] and s['t114_saved_ci']==source_summary['t75']['protocol_aligned_bootstrap_ci'] and s['t114_noise_seeds']==rr['noise_seeds']
 result=dict(complete=True,source_result_commit='cb980799',pins={n:sha(P/n) for n in ['results.json','all_budgets.json','ALL_BUDGETS.md','protocol.json','manifest.json']},checker_sha256=sha(Path(__file__)),report=d['report'],duplicate_replacements=dupcounts,selected_raw_keys={k:len(v) for k,v in selected.items()},arc_cumulants=next(r for r in inventory if r['name']=='arc_cumulants'),ratios=dict(e44=float(r44),t114_own=float(own),t114_uniform=float(uniform)),author_cpu=d['cpu_seconds'],review_cpu=time.process_time()-t,new_author_model_MC_native_calls=0,limitations=['All reported intervals and truth values are inherited; no new uncertainty calculation.','Latest-key history preserved; archival presence is not runtime-source proof.','Two panels and cost conventions remain distinct; missing E44 per-copy costs were not reconstructed.','Classification and determinism policy adoption remain with Claude DRI; no full-cost or rung validation.'])
 (H/'t204_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,rows=498,review_cpu=result['review_cpu'])))
if __name__=='__main__':main()
