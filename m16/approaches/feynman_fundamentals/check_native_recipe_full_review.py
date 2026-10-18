import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from decimal import Decimal as D,localcontext
import gzip,hashlib,json,time,math
import numpy as np
P=Path('/workspace/work/recover-small_h2_910/review/recipe112_native_cost');start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b):assert math.isclose(a,b,rel_tol=8e-15,abs_tol=1e-22),(a,b)
summary=read(P/'panel_summary.json');assert summary['complete'] and len(summary['groups'])==5 and summary['sampler']==351
source_sha=sha(P.parent/'recipe112_recovery/dependencies/understand_m43/est75.py');trace_sha=sha(P/'trace.py');driver_sha=sha(P/'panel.py')
allnumeric=allinteger=0;cpu=0.;ratios=[];cis=[]
for zoo in range(5):
 rows=[]
 for noise in range(80,84):
  stem=f'H8_z{zoo}_n{noise}';d=read(P/'panel'/f'{stem}.json');archive=P/'panel'/d['archive'];assert sha(archive)==d['archive_sha256'];records=[json.loads(x) for x in gzip.decompress(archive.read_bytes()).splitlines()];priorpath=P.parent/'recipe112_recovery/direct_counts'/f'core_{stem}.json';assert sha(priorpath)==d['prior_sha256'];prior=read(priorpath)
  assert d['complete'] and (d['zoo'],d['noise'])==(zoo,noise) and len(d['rows'])==len(records)==64
  assert d['source_sha256']==source_sha and d['trace_sha256']==trace_sha and d['driver_sha256']==driver_sha and d['sampler_numeric']==351
  assert d['checkpoint_sha256']==prior['checkpoint_sha256']
  for i,(r,a,p) in enumerate(zip(d['rows'],records,prior['rows'])):
   assert i==r['copy']==a['copy']==p['copy'] and r['model_sha256']==a['model_sha256']==p['model_sha256'];b,t=a['baseline'],a['traced'];assert all(b[k]==t[k] for k in ['hex','legacy','stats','info']);assert r['estimate']==b['value']==r['saved_estimate']==p['saved_estimate'];assert r['truth']==p['truth'] and r['saved_scalar_ops']==p['ops'];assert r['legacy']==b['legacy']==p['saved_ops'] and r['cells']==b['stats']['cells']==p['saved_cells'];c=Counter()
   for site in t['ledger']['sites'].values():c.update(site)
   assert dict(c)==r['totals']==t['ledger']['totals'];integer=sum(v for k,v in c.items() if k.startswith('integer_control'));numeric=sum(c.values())-integer;assert integer==r['integer']==t['ledger']['integer_control_total'];assert numeric==r['ops']==t['ledger']['named_numeric_total'];allnumeric+=numeric;allinteger+=integer
  rows+=d['rows'];cpu+=d['cpu']
 g=next(g for g in summary['groups'] if g['zoo']==zoo);assert len(rows)==g['models']==256
 with localcontext() as ctx:
  ctx.prec=65;errors=[D(r['estimate'])-D(r['truth']) for r in rows];variance=[D(r['truth'])*(1-D(r['truth'])) for r in rows];ops=[r['ops'] for r in rows];samples=[o//351 for o in ops];mse=sum(e*e for e in errors)/256;den=sum(v/n for v,n in zip(variance,samples))/256;cont=sum(v*351/o for v,o in zip(variance,ops))/256;meanbudget=sum(variance)/256*351/(D(sum(ops))/256)
  for key,value in {'mse':mse,'mean_sampling_variance_integer':den,'ratio_integer_budget':mse/den,'ratio_continuous_budget':mse/cont,'ratio_mean_budget':mse/meanbudget}.items():near(float(value),g[key])
  near(sum(ops)/256,g['mean_ops']);assert min(samples)==g['min_samples'] and max(samples)==g['max_samples']
  e2=np.array([float(e*e) for e in errors]);v=np.array([float(v/n) for v,n in zip(variance,samples)]);indices=np.random.default_rng(0).integers(0,256,(1000,256));boot=e2[indices].sum(1)/v[indices].sum(1);ci=np.quantile(boot,[.025,.975]);assert np.allclose(ci,g['bootstrap95_integer'],rtol=8e-15,atol=1e-22);ratios.append(float(mse/den));cis.append(float(ci[1]))
near(cpu,summary['cpu_replay']);near(max(ratios),summary['worst_ratio']);near(max(cis),summary['largest_conditional_ci_upper'])
out=dict(cases=1280,complete_site_ledgers=1280,shards=20,named_numeric_total=allnumeric,separate_integer_total=allinteger,native_budget_ratios=ratios,largest_conditional_bootstrap_upper=max(cis),sampler_numeric=351,author_replay_cpu=cpu,review_cpu=time.process_time()-start,scope='All saved site ledgers/pins/parity and65-digit per-model-budget contractions checked; no trace/model rerun. Model hashes linked to independently regenerated T113 panel. Conditional bootstrap, named-source units only; no hardware/setup/library/control closure or other H rung inference.')
Path(__file__).with_name('native_recipe_full_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
