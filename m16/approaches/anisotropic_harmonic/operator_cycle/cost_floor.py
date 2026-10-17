"""Conditional product floor; not a total-cost or sampling-win certificate."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'ensemble.json').read_text())
out=dict(rows=[],interpretation=(
    'One scalar product for each contraction multiply-add; additions, weights, angular '
    'assembly, elementary functions, setup and geometry omitted. Sampling denominators '
    'are conventional network-forward operations excluding input generation. A floor '
    'below1 is inconclusive, not a measured estimator win.'))
for row in data['rows']:
    H=int(row['family'][1:]);p=row['reference']
    conventional=2*(3*H*(H+1)+3*H)
    r=dict(family=row['family'],copy=row['copy'],conventional_forward_ops=conventional)
    for method in ('dense','real_operator'):
        w=row[method]['stats']['cut_work']
        lower=sum(w.get(k,0) for k in ('real_gram_multiply_adds','complex_gram_multiply_adds',
                                      'complex_matvec_multiply_adds','real_operator_multiply_adds'))
        r[method]=dict(multiplication_floor=lower,
                      conventional_ratio_at_floor=row[method]['error']**2*lower/(conventional*p*(1-p)))
    out['rows'].append(r)
out['families']={}
for family in ('H2','H4'):
    rows=[r for r in out['rows'] if r['family']==family]
    out['families'][family]={method:dict(
        mean_conventional_ratio_at_floor=float(np.mean([r[method]['conventional_ratio_at_floor'] for r in rows])),
        conventional_above_one=sum(r[method]['conventional_ratio_at_floor']>1 for r in rows),
        minimum_products=min(r[method]['multiplication_floor'] for r in rows),
        maximum_products=max(r[method]['multiplication_floor'] for r in rows))
        for method in ('dense','real_operator')}
(HERE/'cost_floor.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['families']))
