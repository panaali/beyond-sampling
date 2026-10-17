"""Post-hoc stabilization on cached matrices; no new model evaluations."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[k]='1'
import json
import time
from pathlib import Path
import numpy as np
from probe import stats
p=Path(__file__).resolve().parent
start=time.process_time()
raw=json.loads((p/'results.json').read_text())
data=np.load(p/'matrices.npz')
output=[]
for idx,case in enumerate(raw['cases']):
    key=f'm{idx//2}_s{idx%2}'
    A=data[key+'_mass']; cells=data[key+'_cells']
    u,s,vt=np.linalg.svd(A[:16,:16],full_matrices=False)
    curves=[]
    for rank in (2,4,8):
        inv=(vt[:rank].T/s[:rank]) @ u[:,:rank].T
        pred=A[:,:16] @ inv @ A[:16,:]
        # All entries except heldout block are needed, regardless of rank.
        observed=np.ones_like(A,dtype=bool); observed[16:,16:]=False
        metrics = stats(A[16:,16:],pred[16:,16:])
        metrics['conditional_iid_face_se_same_construction_faces'] = float(
            A[16:,16:].std(ddof=1)/np.sqrt(observed.sum()))
        metrics['absolute_mean_error_over_conditional_iid_se'] = float(
            abs(metrics['predicted_mean']-metrics['true_mean']) /
            max(metrics['conditional_iid_face_se_same_construction_faces'],1e-30))
        algebra_flops_bound = 40*16**3 + 2*32*16**2 + 2*32**2*16
        curves.append(dict(rank=rank,construction_faces=int(observed.sum()),
            algebra_flops_bound=algebra_flops_bound,
            construction_fpe_lower_bound=float(2*cells[observed].sum()+algebra_flops_bound/6000),
            construction_walk_fpe=int(2*cells[observed].sum()),
            heldout=metrics,full_grid=stats(A,pred)))
    output.append(dict(model=case['model'],skeleton=case['skeleton'],curves=curves))
result=dict(posthoc=True,additional_face_evaluations=0,cpu_seconds=time.process_time()-start,cases=output)
(p/'stabilized.json').write_text(json.dumps(result,indent=2))
for c in output:
    print(c['model'],c['skeleton'],[(r['rank'],round(r['heldout']['relative_l2'],4),round(r['heldout']['relative_mean_error'],4),round(r['heldout']['top5_absolute_error_mass'],4)) for r in c['curves']])
