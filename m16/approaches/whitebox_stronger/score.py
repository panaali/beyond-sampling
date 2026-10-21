"""Score local frozen predictions; no estimator execution."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
REFS=HERE.parent/'whitebox_cumulants'/'results'/'references'
records=[]
for p in (HERE/'results').glob('*.npy'):
    meta=json.loads(p.with_suffix('.json').read_text())
    for refp in REFS.glob('*_[01].json'):
        refmeta=json.loads(refp.read_text())
        if refmeta['mlp_seed'] != meta['mlp_seed']: continue
        target=np.load(refp.with_suffix('.npz'))['final_means'].astype(float)
        err=np.load(p)[-1].astype(float)-target
        mse=float(np.mean(err**2))
        records.append(dict(file=p.name,row=refp.stem,method=meta['method'],mse=mse,adjusted=mse*max(.1,meta['cost_fraction']),cost_fraction=meta['cost_fraction'],residual=meta['residual_seconds'],contract_pass=meta['local_contract_pass'],rss=meta['peak_rss_bytes']))
(HERE/'results'/'scores.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
