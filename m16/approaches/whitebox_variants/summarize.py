"""Independent saved-array rescore, matched public-row comparisons; no model runs."""
import hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H/'results'; C=H.parent/'whitebox_cumulants/results';S=H.parent/'whitebox_stronger/results'
manifest=json.loads((H/'sources.json').read_text()); groups={};rows=[]; cpu=0
for p in R.glob('*.json'):
 d=json.loads(p.read_text())
 if isinstance(d,dict):cpu+=max(d.get('worker_cpu_seconds',0),d.get('supervisor_cpu_seconds',0))
for method in manifest:
 selected=[]
 for label in ('mini_0','mini_1','full_0','full_1'):
  p=R/f'{method}_{label}.json'
  if not p.exists() or not p.with_suffix('.npy').exists():continue
  d=json.loads(p.read_text()); refp=C/'references'/f'{label}.npz';ref=np.load(refp)['final_means'].astype(float);y=np.load(p.with_suffix('.npy'))[-1].astype(float)
  mse=float(np.mean((y-ref)**2));score=mse*max(.1,d['cost_fraction']);co=json.loads((C/f'covariance_{label}.json').read_text()); cy=np.load(C/f'covariance_{label}.npy')[-1].astype(float);cs=float(np.mean((cy-ref)**2))*max(.1,co['cost_fraction'])
  row=dict(method=method,row=label,mse=mse,score=score,cost_fraction=d['cost_fraction'],residual=d['residual_seconds'],rss=d['peak_rss_bytes'],contract_pass=d['local_contract_pass'],covariance_score=cs,ratio_to_covariance=score/cs,strict_point_pass=score/cs<=1,lenient_point_pass=score/cs<=10)
  baseline=S/f'e30_{label.replace("_", "")}.json'
  if baseline.exists():
   b=json.loads(baseline.read_text());by=np.load(baseline.with_suffix('.npy'))[-1].astype(float);bs=float(np.mean((by-ref)**2))*max(.1,b['cost_fraction']);row.update(e30_score=bs,ratio_to_e30=score/bs)
  selected.append(row);rows.append(row)
 if not selected:continue
 score=float(np.mean([x['score'] for x in selected]));cov=float(np.mean([x['covariance_score'] for x in selected])); matched=[x for x in selected if 'e30_score' in x]
 groups[method]=dict(n=len(selected),mean_score=score,ratio_to_covariance=score/cov,strict_point_pass=score/cov<=1,lenient_point_pass=score/cov<=10,max_residual=max(x['residual'] for x in selected),max_rss=max(x['rss'] for x in selected),source_sha256=manifest[method]['sha256'])
 if matched:groups[method].update(e30_matched_n=len(matched),matched_ratio_to_e30=sum(x['score'] for x in matched)/sum(x['e30_score'] for x in matched))
report=dict(groups=groups,rows=rows,total_recorded_worker_cpu=cpu,scope='All four public reference rows treated as DEV; no population CI or official rank; failed guard is included in CPU. Point bars relative to covariance, not AlgZoo F.')
(R/'aggregate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(groups,indent=2));print('CPU',cpu)
