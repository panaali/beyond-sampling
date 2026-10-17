"""Conservative dense-multiplication floor, with explicit sampling conventions."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'ensemble_results.json').read_text())
out=dict(rows=[],interpretation=(
    'Each executed dense real or complex multiply-add contains at least one scalar multiplication. '
    'This omits additions, other products, special functions, setup and all geometry. '
    'Sampling denominators are stated network-forward conventions, excluding input generation; '
    'ratios are conditional comparisons, not an unconditional sampling-impossibility proof.'))
for row in data['rows']:
    H=int(row['family'][1:]);stats=row['lune']['stats'];w=stats['cut_work']
    lower=sum(w.get(k,0) for k in ('real_gram_multiply_adds','complex_gram_multiply_adds','complex_matvec_multiply_adds'))
    p=row['reference'];error=row['lune']['error']
    affine=3*(2*H*H+3*H-1)
    conventional=2*(3*H*(H+1)+3*H)
    out['rows'].append(dict(family=row['family'],copy=row['copy'],dense_multiplication_floor=lower,
        affine_only_forward_ops=affine,conventional_forward_ops=conventional,
        affine_ratio_at_floor=error**2*lower/(affine*p*(1-p)),
        conventional_ratio_at_floor=error**2*lower/(conventional*p*(1-p))))
out['families']={}
for family in ('H2','H4'):
    rows=[r for r in out['rows'] if r['family']==family]
    out['families'][family]=dict(
        mean_affine_ratio_at_floor=float(np.mean([r['affine_ratio_at_floor'] for r in rows])),
        mean_conventional_ratio_at_floor=float(np.mean([r['conventional_ratio_at_floor'] for r in rows])),
        conventional_above_one=sum(r['conventional_ratio_at_floor']>1 for r in rows))
(HERE/'cost_floor_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['families']))
