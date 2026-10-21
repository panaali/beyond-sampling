"""Frozen robustness check on public full rows2/3; no fitting or selection changes."""
import json,subprocess,sys,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;refs=Path(sys.argv[1]); records=[]
for label in ('full_2','full_3'):
 row=json.loads((refs/f'{label}.json').read_text()); truth=np.load(refs/f'{label}.npz')['final_means'].astype(float)
 for method in ('e30_rank448','e30_tau30'):
  if time.strftime('%H:%M',time.gmtime())>='23:57':raise RuntimeError('No new compute after23:57UTC')
  p=H/'results'/f'{method}_{label}.json'
  if not p.exists():subprocess.run([sys.executable,str(H/'run.py'),'--method',method,'--width','1024','--seed',str(row['mlp_seed']),'--output',str(p)],check=True)
  d=json.loads(p.read_text()); y=np.load(p.with_suffix('.npy'))[-1].astype(float);mse=float(np.mean((y-truth)**2));score=mse*max(.1,d['cost_fraction'])
  r=dict(method=method,row=label,mse=mse,adjusted=score,cost_fraction=d['cost_fraction'],residual=d['residual_seconds'],rss=d['peak_rss_bytes'],contract_pass=d['local_contract_pass'],source_sha256=d['source_sha256']);records.append(r);print(json.dumps(r),flush=True)
  (H/'results/fresh_scores.json').write_text(json.dumps(records,indent=2)+'\n')
