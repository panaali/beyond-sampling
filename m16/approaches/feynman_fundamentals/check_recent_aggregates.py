"""Independent Decimal contraction of saved T67 and T40 rows; no engines."""
import json
from decimal import Decimal as D, getcontext
from pathlib import Path
getcontext().prec=60
HERE=Path(__file__).resolve().parent
root=Path('/workspace/work')
p=root/'recover-small_h4_4/m16/approaches/rank_conditioned_transport/penultimate_cycle/pilot.json'
data=json.loads(p.read_text(),parse_float=D)
rows=[]
for row in data['rows']:
    total=sum(row['stats']['contributions'].values(),D(0))
    error=total-data['reference']
    assert abs(total-row['estimate'])<D('2e-16')
    assert abs(error-sum(row['per_label_errors'],D(0)))<D('2e-16')
    rows.append(dict(order=row['order'],cached=row['cached'],total=str(total),error=str(error)))
p=root/'recover-small_h2_910/m16/approaches/endpoint_cumulant_review/support_conditioned.json'
s=json.loads(p.read_text(),parse_float=D)
totals={}
for method in ['orthant','component_cone']:
    value=s['constant_correct_mass']
    for row in s['rows']:
        value+=row['mass']*row['methods'][method]['fine']['q'][row['target']]
    assert abs(value-s['accuracy'][method])<D('2e-16')
    totals[method]=dict(value=str(value),bias=str(value-s['truth']))
out=dict(penultimate=rows,support_conditioned=totals,
         support_rank2_components=len(s['rows']),rank1_components=len(s['rank_one_controls']),
         scope='Saved-row arithmetic only; independent of author aggregation code, shared numerical inputs.')
(HERE/'recent_aggregate_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
