import json,time,hashlib
from pathlib import Path
from fractions import Fraction as F
start=time.process_time()
source=Path('/workspace/work/recover-small_h2_910/m16/approaches/walkfree_gain_review/associations.json')
data=json.loads(source.read_text());groups={}
for r in data['joined_model_rows']:groups.setdefault((r['H'],r['T']),[]).append(r)
assert len(groups)==18 and all(len(v)==5 for v in groups.values())
rows=[]
for feature in ('gate_uncertainty','boundary_alignment'):
 for gain in ('sobol','iid'):
  per=[]
  for key,group in sorted(groups.items()):
   x=[r[feature] for r in group];y=[r[gain+'_gain'] for r in group]
   assert len(set(x))==len(set(y))==5
   rx=[sum(a<b for a in x)-2 for b in x];ry=[sum(a<b for a in y)-2 for b in y]
   rho=F(sum(a*b for a,b in zip(rx,ry)),10)
   per.append(dict(H=key[0],T=key[1],rho=str(rho)))
  mean=sum((F(r['rho']) for r in per),F(0))/len(per)
  expected=next(r['rho'] for rep in data['reports'] if rep['subset']=='all90' for r in rep['results'] if r['conditioning']=='within_HT' and r['feature']==feature and r['gain']==gain)
  rows.append(dict(feature=feature,gain=gain,group_rhos=per,exact_mean_rho=str(mean),mean_rho=float(mean),saved_rho=expected,difference=float(mean)-expected))
out=dict(results=rows,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cpu_seconds=time.process_time()-start)
Path(__file__).with_name('gain_association_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='group_rhos'} for r in rows],indent=2))
