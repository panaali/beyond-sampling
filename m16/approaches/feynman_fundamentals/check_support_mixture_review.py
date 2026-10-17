"""Independent saved-component aggregation, no endpoint/geometry imports."""
import json,time
from decimal import Decimal,localcontext
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/recover-small_h2_910/m16/approaches/endpoint_cumulant_review')
start=time.process_time()
mix=json.loads((SOURCE/'gate_mixture.json').read_text(),parse_float=Decimal)
gauss=json.loads((SOURCE/'full_covariance_bounded.json').read_text(),parse_float=Decimal)
with localcontext() as ctx:
    ctx.prec=60
    fixed=sum((x['mass'] for x in mix['components'] if x['fixed_winner'] is not None),Decimal(0))
    fixed_correct=sum((x['mass'] for x in mix['components'] if x['fixed_winner']==x['target']),Decimal(0))
    values={}
    for kind in mix['accuracy']:
        total=fixed_correct
        for x in mix['components']:
            if x['fixed_winner'] is None:total+=x['mass']*x['endpoint']['q_raw'][kind][x['target']]
        assert abs(total-mix['accuracy'][kind])<Decimal('3e-15')
        values[kind]=str(total)
    by_key={(x['gate_cell'],tuple(x['order'])):x for x in mix['components']}
    total=fixed_correct
    for x in gauss['mixture']['cases']:
        original=by_key[x['gate_cell'],tuple(x['order'])]
        assert x['mass']==original['mass'] and original['fixed_winner'] is None
        total+=x['mass']*x['fine']['q'][original['target']]
    assert abs(total-gauss['mixture']['accuracy'])<Decimal('3e-15')
    result=dict(components=len(mix['components']),constant_components=sum(x['fixed_winner'] is not None for x in mix['components']),
                total_mass=str(sum((x['mass'] for x in mix['components']),Decimal(0))),fixed_mass=str(fixed),
                fixed_correct_mass=str(fixed_correct),raw_accuracies=values,full_covariance_accuracy=str(total),
                saved_full_covariance_accuracy=str(gauss['mixture']['accuracy']),cpu_seconds=time.process_time()-start,
                scope='Independent Decimal aggregation of saved probabilities and masses, not a new endpoint or geometric certificate.')
(HERE/'support_mixture_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
