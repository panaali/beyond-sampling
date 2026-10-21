"""Fresh-row scoring only; estimator settings frozen before these references."""
import json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
rows=[]
for index in (2,3):
 refpath=HERE/'results'/'fresh_references'/f'full_{index}.npz'
 ref=np.load(refpath)['final_means'].astype(float)
 meta=json.loads(refpath.with_suffix('.json').read_text())
 result=HERE/'results'/f'e30_full{index}.json'
 if not result.exists(): continue
 run=json.loads(result.read_text())
 if 'mlp_seed' not in run: continue
 assert meta['mlp_seed']==run['mlp_seed']
 pred=np.load(result.with_suffix('.npy'))[-1].astype(float)
 mse=float(np.mean((pred-ref)**2))
 rows.append(dict(row=f'full_{index}',seed=meta['mlp_seed'],raw_mse=mse,adjusted_error=mse*max(.1,run['cost_fraction']),cost_fraction=run['cost_fraction'],local_contract_pass=run['local_contract_pass'],residual_seconds=run['residual_seconds'],rss_bytes=run['peak_rss_bytes'],reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),source_sha256=run['source_sha256'],scope='Fresh to our development only; upstream public training overlap possible; no fitting'))
(HERE/'results'/'fresh_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
