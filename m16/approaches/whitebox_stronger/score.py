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
        unpenalized=mse*max(.1,meta['cost_fraction'])
        adjusted=unpenalized if meta['local_contract_pass'] else float(np.mean(target**2))
        prior=json.loads((HERE.parent/'whitebox_cumulants'/'results'/'summary.json').read_text())
        cov=next(r['methods']['covariance']['official_formula_score'] for r in prior['records'] if r['row']==refp.stem)
        ratio=adjusted/cov
        records.append(dict(file=p.name,row=refp.stem,method=meta['method'],mse=mse,adjusted=adjusted,unpenalized_adjusted=unpenalized,cost_fraction=meta['cost_fraction'],residual=meta['residual_seconds'],contract_pass=meta['local_contract_pass'],rss=meta['peak_rss_bytes'],score_ratio_to_covariance=ratio,strict_point_pass=ratio<=1,lenient_point_pass=ratio<=10))
(HERE/'results'/'scores.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
