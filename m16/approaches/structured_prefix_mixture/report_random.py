"""Contract every frozen random-width cell; no model/variant selection."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;P=json.loads((HERE/'random_protocol.json').read_text())
rows=[];cells=[]
for h in P['widths']:
    for d in (1,h):
        dd=[]
        for seed in P['model_seeds']:
            path=HERE/f'random_H{h}_d{d}_s{seed}.json';r=json.loads(path.read_text());assert r['complete'];a=HERE/f'random_H{h}_d{d}_s{seed}.npz';assert hashlib.sha256(a.read_bytes()).hexdigest()==r['archive_sha256'];dd.append(r);cells.append(r)
        row=dict(H=h,d=d,models=len(dd),mean_vMSE={m:float(np.mean([r['methods'][m]['mean_vMSE'] for r in dd])) for m in dd[0]['methods']},per_seed_vMSE={m:[r['methods'][m]['mean_vMSE'] for r in dd] for m in dd[0]['methods']},method_cpu_seconds={m:sum(r['methods'][m]['cpu_seconds'] for r in dd) for m in dd[0]['methods']})
        if d==1:
            row['mixture_vs_gauss_ratio']=row['mean_vMSE']['sector_mixture']/row['mean_vMSE']['gauss'];row['mixture_vs_collapsed_ratio']=row['mean_vMSE']['sector_mixture']/row['mean_vMSE']['collapsed_h2'];row['component_counts']=[r['methods']['sector_mixture']['components'] for r in dd]
        rows.append(row)
scalar=[r for r in cells if r['d']==1];out=dict(complete=True,cells=len(cells),rows=rows,scalar_mixture_better_than_gauss=sum(r['methods']['sector_mixture']['mean_vMSE']<r['methods']['gauss']['mean_vMSE'] for r in scalar),scalar_mixture_worse_than_collapsed=[dict(H=r['H'],seed=r['model_seed']) for r in scalar if r['methods']['sector_mixture']['mean_vMSE']>r['methods']['collapsed_h2']['mean_vMSE']],reference_noise_floor=1/P['reference_samples'],reference_gaussian_scalars=sum(r['compute_accounting']['reference_gaussian_scalars'] for r in cells),cell_cpu_seconds=sum(r['cpu_seconds'] for r in cells),reference_cpu_seconds=sum(r['reference_cpu_seconds'] for r in cells),chunk_cpu_seconds=sum(json.loads((HERE/f'random_H{h}_completion.json').read_text())['cpu_seconds'] for h in P['widths']),qualification='Finite24-cell diagnostic,3 model seeds/width. No asymptotic monotonicity, final accuracy, K>2, all-closure exhaustion or equal-compute claim. Means contain reference noise, floor1/N shown; floor-subtracted cell values preserved without clipping.')
(HERE/'random_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out))
