import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from decimal import Decimal as D,localcontext
import hashlib,json,math,time,numpy as np
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery');start=time.process_time();audit=json.loads((P/'audit.json').read_text());manifest=json.loads((P/'manifest.json').read_text())
for name,r in manifest.items():
 if '/' not in name:assert hashlib.sha256((P/'frozen'/name).read_bytes()).hexdigest()==r['sha256']
checked=0;clean=0;ratios=[];upper=[];coverage={}
with localcontext() as ctx:
 ctx.prec=70
 for k in (28,30):
  data=json.loads((P/'frozen'/f'margin_k{k}.json').read_text());rung=json.loads((P/'frozen'/f'rung_k{k}.json').read_text());coverage[k]=list(data['seeds'])
  for seed,row in data['seeds'].items():
   for recipe,c in row['clean'].items():
    t=D(rung['seeds'][seed]['clean']['truth']);ratio=D(c['err'])**2*D(c['ops'])/(72*t*(1-t));assert math.isclose(float(ratio),c['ratio'],rel_tol=5e-14);clean+=1
   for panel in ('p20','p100'):
    for recipe,r in row.get(panel,{}).items():
     e=list(map(D,r['errs']));n=len(e);mse=sum(x*x for x in e)/n;bias=sum(e)/n;assert math.isclose(math.sqrt(float(mse)),r['rms'],rel_tol=5e-14);assert math.isclose(float(bias),r['bias'],rel_tol=5e-14,abs_tol=1e-20);assert max(map(abs,r['errs']))==r['max_abs'];checked+=1
     if panel=='p20':
      tr=list(map(D,rung['seeds'][seed]['panel']['truths']));pq=sum(t*(1-t) for t in tr)/n;ratio=float(mse*D(r['ops'])/(72*pq));assert math.isclose(ratio,r['ratio'],rel_tol=5e-14);ratios.append(ratio)
      # Independent sequential bootstrap loop, same declared RNG law and fixed mean cost.
      rng=np.random.default_rng(0);ee=np.array(r['errs']);tt=np.array(list(map(float,tr)));samples=[]
      for _ in range(1000):
       ix=rng.integers(0,n,n);samples.append(float(np.mean(ee[ix]**2)/np.mean(tt[ix]*(1-tt[ix]))*r['ops']/72))
      ci=np.quantile(samples,[.025,.975]);assert np.allclose(ci,r['ci'],rtol=5e-14,atol=1e-20);upper.append(float(ci[1]))
assert checked==15 and clean==12 and len(ratios)==12 and len(coverage[28])==5 and coverage[30]==['0']
out=dict(panel_rows=checked,p20_rows=len(ratios),p100_error_stat_rows=3,clean_rows=clean,coverage=coverage,worst_p20_legacy_ratio=max(ratios),worst_p20_bootstrap_upper=max(upper),review_cpu=time.process_time()-start,scope='Saved error statistics and original72 fixed-mean-cost bootstrap only. p100 per-copy truths/counters absent; no independent p100 ratio, truth validation, full common-unit cost or broader rung claim.')
Path(__file__).with_name('margin_training_recovery_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
