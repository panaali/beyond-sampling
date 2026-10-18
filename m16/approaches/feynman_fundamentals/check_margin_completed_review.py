import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time,numpy as np,torch
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery');start=time.process_time();s=json.loads((P/'continuation_summary.json').read_text());pilot=json.loads((P/'continuation_pilot.json').read_text());prior={(r['seed'],r['index']):r for r in pilot['rows']};manifest=json.loads((P/'continuation_manifest.json').read_text());rung=json.loads((P/'frozen/rung_k30.json').read_text())['seeds'];cpu=0;total=0;losses=[];worst=0
for seed in range(1,5):
 path=Path(manifest[f'checkpoint_{seed}']['source']);assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[f'checkpoint_{seed}']['sha256'];st=torch.load(path,map_location='cpu',weights_only=True);base=tuple(x.numpy().astype(float) for x in [st['rnn.weight_ih_l0'][:,0],st['rnn.weight_hh_l0'],st['linear.weight']]);rows=[]
 for ns in range(20,24):
  d=json.loads((P/f'k30_s{seed}_noise{ns}.json').read_text());assert d['complete'] and len(d['rows'])==256;cpu+=d['cpu_seconds'];rng=np.random.default_rng(ns)
  for j,r in enumerate(d['rows']):
   ix=(ns-20)*256+j;assert r['seed']==seed and r['index']==ix and r['saved_truth']==rung[str(seed)]['panel']['truths'][ix];model=tuple(w+.05*rng.standard_normal(w.shape) for w in base);assert hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest()==r['model_hash']
   if j<4:assert r['reused_pilot'] and {k:v for k,v in r.items() if k!='reused_pilot'}==prior[seed,ix]
   else:assert not r.get('reused_pilot',False)
   rows.append(r)
 total+=len(rows)
 for recipe in ['r85b','r112']:
  q=next(v for v in s['panels'] if v['seed']==seed and v['recipe']==recipe);truth=np.array([r['saved_truth'] for r in rows]);err=np.array([r['recipes'][recipe]['value'] for r in rows])-truth;op=np.array([r['recipes'][recipe]['native_ops'] for r in rows]);pq=truth*(1-truth);ef=list(map(F,err));tf=list(map(F,truth));mse=sum(e*e for e in ef)/1024;v=sum(t*(1-t) for t in tf)/1024;meanop=F(int(sum(op)),1024);ratio=mse*meanop/(72*v);assert math.isclose(float(ratio),q['statistics']['original72_ratio'],rel_tol=2e-14);assert math.isclose(float(sum(ef)/1024),q['statistics']['bias'],rel_tol=2e-14,abs_tol=1e-20)
  matched=float(mse/(sum(72*t*(1-t)/int(o) for t,o in zip(tf,op))/1024));assert math.isclose(matched,q['per_copy_budget_ratio'],rel_tol=2e-14);ix=np.random.default_rng(0).integers(0,1024,(1000,1024));e2=err**2
  fixed=e2[ix].sum(1)/pq[ix].sum(1)*(float(meanop)/72);uniform=e2[ix].mean(1)*op[ix].mean(1)/(72*pq[ix].mean(1));separate=e2[ix].mean(1)/(72*pq/op)[ix].mean(1)
  for values,key in [(fixed,'original_fixed_cost_bootstrap'),(uniform,'paired_cost_uniform_budget_ci'),(separate,'paired_cost_per_copy_budget_ci')]:assert np.allclose(np.quantile(values,[.025,.975]),q[key],rtol=3e-14,atol=1e-20)
  loss=int(np.sum(e2>=72*pq/op));maximum=float(np.max(e2/(72*pq/op)));assert loss==q['copy_ratio_ge1'] and maximum==q['worst_copy_ratio'];losses.append(dict(seed=seed,recipe=recipe,individual_losses=loss,worst_copy_ratio=maximum));worst=max(worst,float(ratio))
for c in s['clean']:
 r=prior[c['seed'],-1];z=r['recipes'][c['recipe']];p=F(r['saved_truth']);e=F(z['value']-r['saved_truth']);ratio=e*e*z['native_ops']/(72*p*(1-p));assert math.isclose(float(ratio),c['statistics']['original72_ratio'],rel_tol=2e-14,abs_tol=1e-25)
assert total==4096 and len(s['panels'])==8 and len(s['clean'])==8 and math.isclose(cpu,s['shard_cpu'],rel_tol=1e-14)
out=dict(noisy_models=total,recipe_panels=8,clean_rows=8,pilot_rows_reused_once=64,all_model_hashes_reconstructed=True,worst_legacy72_aggregate_ratio=worst,individual_losses=losses,shard_cpu=cpu,review_cpu=time.process_time()-start,scope='Saved historical72/native convention, conditional pointwise bootstrap, fixed panel. No estimator/reference run, simultaneous ladder or full native-cost guarantee; individual losses retained.')
Path(__file__).with_name('margin_completed_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
