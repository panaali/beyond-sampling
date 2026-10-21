import json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REFS=HERE.parent/'whitebox_cumulants/results/references'
records=[]
for name in sys.argv[1:] or ['block256','rank320','age3','corrected']:
 for label in ['mini_0','mini_1','full_0','full_1']:
  refmeta=json.loads((REFS/f'{label}.json').read_text())
  p=HERE/'results'/f'{name}_{label}.json'
  if not p.exists():
   subprocess.run([sys.executable,str(HERE/'run.py'),'--method',name,'--width','1024','--seed',str(refmeta['mlp_seed']),'--output',str(p)],check=True)
  meta=json.loads(p.read_text());pred=np.load(p.with_suffix('.npy'))[-1].astype(float);ref=np.load(REFS/f'{label}.npz')['final_means'].astype(float)
  mse=float(np.mean((pred-ref)**2));score=mse*max(.1,meta['cost_fraction'])
  r={'method':name,'row':label,'raw_mse':mse,'adjusted_error':score,'residual_seconds':meta['residual_seconds'],'cost_fraction':meta['cost_fraction'],'rss_bytes':meta['peak_rss_bytes'],'local_contract_pass':meta['local_contract_pass'],'cpu':meta['worker_cpu_seconds']}
  records.append(r);print(json.dumps(r),flush=True)
 (HERE/'results'/f'scores_{name}.json').write_text(json.dumps(records[-4:],indent=2)+'\n')
