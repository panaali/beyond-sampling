"""Independent exact saved T4 point-ratio and partial-group contraction."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time
HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h4_4/review/split_attribution_review/t4_increment')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(v):assert type(v) in (int,float) and math.isfinite(v);return F(v)
def equal(r,value):
 assert F(int(r['numerator']),int(r['denominator']))==value and r['float']==float(value)
def enc(v):return dict(numerator=str(v.numerator),denominator=str(v.denominator),float=float(v))
def main():
 start=time.process_time();dest=HERE/'t198_t4_saved_review.json';assert not dest.exists()
 r=read(BASE/'results.json');g=read(BASE/'execution_gate.json');m=read(BASE/'manifest.json')
 assert r['status']=='pass' and r['gate_sha256']==sha(BASE/'execution_gate.json') and r['manifest_sha256']==sha(BASE/'manifest.json') and r['checker_sha256']==sha(BASE/'check_saved.py')
 for path,h in g['pins'].items():assert sha(path)==h,path
 source=read(g['source_review']);assert source['complete'] and source['pass_all']
 oldreview=read(g['prior_T3_review']);assert oldreview['complete'] and oldreview['pass_all'] and oldreview['pins']['saved_results.json']==sha(g['prior_T3_results'])
 old=read(g['prior_T3_results']);oldrows={tuple(v['key']):v for v in old['rows']};assert len(oldrows)==51
 for row in m['records']:
  file=BASE/'frozen'/row['original'];assert sha(file)==row['sha256'] and file.stat().st_size==row['bytes']
 truth=read(BASE/'frozen/m16/approaches/estT_fix/summary.json')
 expected={(0,t,x) for t in ['cap','mass'] for x in [1e-4,1e-5,1e-6,1e-7,1e-8]}|{(1,'cap',x) for x in [1e-4,1e-5,1e-6]}|{(1,'mass',x) for x in [1e-4,1e-5]}
 assert len(r['rows'])==r['rows_checked']==15 and {tuple(v['key']) for v in r['rows']}==expected
 groups={};maximum={'cap':F(0),'mass':F(0)};maxbound=maximum.copy();maxres=F(0)
 for row in r['rows']:
  file=BASE/'frozen'/row['name'];a=read(file);s,t,tau=row['key'];assert row['sha256']==sha(file) and (a['seed'],a['test'],a['tau'])==(s,t,tau) and (a['H'],a['T'])==(8,4)
  p=q(truth[f'zoo{s}']['exact']);err=q(a['acc'])-p;bound=q(a['bound']);flops=a['flops'];assert flops==2*a['ops'] and flops>0 and 0<p<1
  variance=p*(1-p)*682/flops;ratio=err*err/variance;br=bound*bound/variance
  oldN=q(a['N'])-F(3*flops,2000);res=q(a['bound'])-sum((q(v[1]) for v in a['by_step'].values()),F())-(q(1e-15)*a['pruned'] if t=='mass' else F())
  for field,value in [('error',err),('bound',bound),('continuous_draws',F(flops,682)),('variance',variance),('ratio_actual',ratio),('ratio_bound',br),('historical_N_residual',oldN),('padded_step_residual',res)]:equal(row[field],value)
  assert abs(oldN)<=2*q(math.ulp(a['N'])) and sum(v[0] for v in a['by_step'].values())==a['pruned'] and a['wrong_centroid']<=a['pruned']
  prior=oldrows[s,t,tau];ref=read(BASE/f'frozen/m16/approaches/split_attribution/ref/H8_T4_s{s}.json')
  assert row['T3_record_sha256']==prior['sha256'] and row['T3_cells_same_seed']==prior['retained_cells']
  assert row['cells']==a['retained_cells'] and row['unpruned_cells']==ref['cells'] and row['flops']==flops and row['author_wall_seconds']==a['sec']
  groups.setdefault((t,tau),[]).append((s,ratio,br,row['cells'],row['T3_cells_same_seed']))
  maximum[t]=max(maximum[t],ratio);maxbound[t]=max(maxbound[t],br);maxres=max(maxres,abs(res))
 assert len(r['groups'])==r['groups_checked']==len(groups)==10
 growth=[]
 for row in r['groups']:
  values=groups[row['test'],row['tau']];n=len(values);seeds=sorted(x[0] for x in values)
  assert row['seeds']==seeds and row['missing_seeds']==sorted(set(range(5))-set(seeds))
  ratio=F(sum(x[3] for x in values),sum(x[4] for x in values));growth.append(ratio)
  for field,value in [('mean_cells_available',F(sum(x[3] for x in values),n)),('max_ratio_actual_available',max(x[1] for x in values)),('max_ratio_bound_available',max(x[2] for x in values)),('T4_over_T3_cells_matched_seeds',ratio)]:equal(row[field],value)
 events=list(map(json.loads,(BASE/'progress.jsonl').read_text().splitlines()))
 assert events==[dict(phase='started',gate_sha256=sha(BASE/'execution_gate.json'))]+[dict(phase='row_checked',name=v['name']) for v in r['rows']]+[dict(phase='completed',result_sha256=sha(BASE/'results.json'))]
 assert r['prior_T3_rows_reused']==15 and r['prior_T3_results_sha256']==sha(g['prior_T3_results']) and r['T5_archive_loaded'] is False and r['author_calls']==0
 assert r['author_wall_seconds']==sum(x['author_wall_seconds'] for x in r['rows'])
 result=dict(complete=True,pass_all=True,task='T198 independent incremental T4 saved audit',raw_commit='e5a37806',pins={str(BASE/n):sha(BASE/n) for n in ['results.json','progress.jsonl','execution_gate.json','manifest.json','check_saved.py']},checker_sha256=sha(__file__),rows=15,partial_groups=10,progress_events=len(events),prior_T3_rows_reused=15,max_point_ratio={k:enc(v) for k,v in maximum.items()},max_reported_bound_squared_ratio={k:enc(v) for k,v in maxbound.items()},matched_seed_cell_growth_range=list(map(enc,[min(growth),max(growth)])),max_padded_step_residual=enc(maxres),review_cpu_seconds=time.process_time()-start,prior_checker_cpu_seconds=r['review_cpu'],new_author_calls=0,scope='Exact stored-float arithmetic and partial matched-seed summaries only. No universal/outward cap certificate, actual-pointed-cell failure, total-estimator error bound, full cost, CI, integer sampling budget, T5 or complete-five-seed rung claim. All large ratios retained; reported B is not promoted to a certified bound.')
 dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,sha256=sha(dest),max_point={k:float(v) for k,v in maximum.items()},cpu=result['review_cpu_seconds'])))
if __name__=='__main__':main()
