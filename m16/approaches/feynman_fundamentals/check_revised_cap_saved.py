"""Independent exact saved pruning ratios. No cap/reference/author execution."""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict
import json,hashlib,time,math
P=Path('/workspace/work/recover-small_h4_4/review/split_attribution_review/revised_cap');H=Path(__file__).resolve().parent
PREFIX='m16/approaches/split_attribution/'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def enc(v):return dict(numerator=str(v.numerator),denominator=str(v.denominator),float=float(v))
def main():
 start=time.process_time();r=read(P/'saved_results.json');m=read(P/'manifest.json');g=read(P/'execution_gate.json')
 assert r['status']=='pass' and r['manifest_sha256']==sha(P/'manifest.json') and r['checker_sha256']==sha(P/'check_saved.py') and r['gate_sha256']==sha(P/'execution_gate.json')
 assert g['status']=='completed_source_review' and g['review_commit']=='c22e80a1'
 for n,h in g['pins'].items():assert sha(Path(n) if Path(n).is_absolute() else P/n)==h
 assert len(m['records'])==73
 files={}
 for item in m['records']:
  f=P/'frozen'/item['original'];assert sha(f)==item['sha256'] and f.stat().st_size==item['bytes'];files[item['original']]=f
 refs=read(files[PREFIX+'ref/exact_H8_T3.json']);bygroup=defaultdict(list);keys=set();maxerr=F(0);maxres=F(0);raw_names=sorted(n for n in files if n.startswith(PREFIX+'prune/'))
 assert raw_names==m['prune_paths'] or raw_names==sorted(m['prune_paths'])
 assert len(r['rows'])==len(raw_names)==r['rows_checked']==51
 for saved,name in zip(r['rows'],raw_names,strict=True):
  a=read(files[name]);ref=read(files[PREFIX+f'ref/H8_T3_s{a["seed"]}.json']);key=(a['seed'],a['test'],a['tau']);assert key not in keys;keys.add(key)
  assert a['H']==8 and a['T']==3 and a['flops']==2*a['ops'] and a['wrong_centroid']<=a['pruned']
  assert all(type(a[k]) is int and a[k]>=0 for k in ['ops','flops','retained_cells','resolved','pruned','wrong_centroid'])
  assert sum(v[0] for v in a['by_step'].values())==a['pruned']
  for k,v in a['by_step'].items():assert k in ['1','2','3'] and len(v)==2 and type(v[0]) is int and v[0]>=0
  p=F(refs[str(a['seed'])]);e=F(a['acc'])-p;b=F(a['bound']);n=F(a['flops'],511);variance=p*(1-p)/n
  ra=e*e/variance;rb=b*b/variance;by=sum((F(v[1]) for v in a['by_step'].values()),F(0));pad=F(1e-15)*a['pruned'] if a['test']=='mass' else F(0);res=b-by-pad;hist=F(a['N'])-F(a['flops'],500)
  assert abs(hist)<=F(math.ulp(a['N']))
  expected=dict(name=name,sha256=sha(files[name]),key=list(key),estimate_error=enc(e),bound=enc(b),canonical_continuous_draws=enc(n),historical_N_residual=enc(hist),canonical_variance=enc(variance),ratio_actual=enc(ra),ratio_bound=enc(rb),padded_bound_minus_step_sums=enc(res),retained_cells=a['retained_cells'],reference_cells=ref['cells'],flops=a['flops'],pruned=a['pruned'],author_wall_seconds=a['sec'])
  assert expected==saved
  if a['tau']:bygroup[(a['test'],a['tau'])].append((a['seed'],ra,rb,a['retained_cells'],ref['cells'],a['flops']))
  maxerr=max(maxerr,ra);maxres=max(maxres,abs(res))
 assert keys=={(s,t,q) for s in range(5) for t in ['cap','mass'] for q in [1e-4,1e-5,1e-6,1e-7,1e-8]}|{(0,'cap',0)}
 assert len(r['groups'])==r['groups_checked']==len(bygroup)==10
 caps={}
 for actual,((test,tau),vs) in zip(r['groups'],sorted(bygroup.items()),strict=True):
  assert len(vs)==5 and {v[0] for v in vs}==set(range(5))
  expect=dict(test=test,tau=tau,seeds=[v[0] for v in vs],max_ratio_actual=enc(max(v[1] for v in vs)),max_ratio_bound=enc(max(v[2] for v in vs)),mean_cells=enc(F(sum(v[3] for v in vs),5)),mean_reference_cells=enc(F(sum(v[4] for v in vs),5)),mean_flops=enc(F(sum(v[5] for v in vs),5)))
  assert expect==actual;caps[test]=max(caps.get(test,F(0)),max(v[2] for v in vs))
 events=[json.loads(x) for x in (P/'saved_progress.jsonl').read_text().splitlines()]
 assert events==[dict(event='started',gate_sha256=sha(P/'execution_gate.json'))]+[dict(event='row_checked',name=n) for n in raw_names]+[dict(event='completed',result_sha256=sha(P/'saved_results.json'))]
 text=files[PREFIX+'capcheck.txt'].read_text().splitlines();assert len(text)==136 and text[-1]=='checks 834 violations 0'
 assert r['printed_control_summary']==dict(checks=834,violations=0,retained_lines=135,not_independently_replayed=True,all_rays_and_values_absent=True)
 assert r['T4_T5_replay_files']==0 and r['rendered_report_present'] is False and r['author_cap_geometry_mass_model_RNG_calls']==0
 assert r['author_wall_seconds']==sum(row['author_wall_seconds'] for row in r['rows'])
 out=dict(complete=True,pass_all=True,raw_commit='389142af',pins={n:sha(P/n) for n in ['saved_results.json','saved_progress.jsonl','execution_gate.json','manifest.json','check_saved.py']},checker_sha256=sha(__file__),rows=51,groups=10,payloads=73,progress_events=len(events),max_actual_point_ratio=enc(maxerr),max_reported_bound_ratio={k:enc(v) for k,v in caps.items()},max_step_pad_residual=enc(maxres),review_cpu=time.process_time()-start,prior_saved_cpu=r['review_process_cpu'],new_author_calls=0,scope='Exact stored-float rational comparisons and saved provenance only. No universal/outward cap certificate, independent834 replay, actual pointed-cell failure, T4/T5 evidence, full estimator error bound, CI, integer-budget or complete-cost claim. Printed controls remain partial; fixed pads and pruning-domain premises retain source-review limitations.')
 (H/'revised_cap_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
