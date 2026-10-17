"""Independent Decimal point arithmetic; no author analysis imports/model runs."""
import decimal
import hashlib
import json
import time
from pathlib import Path

START = time.process_time()
decimal.getcontext().prec = 60
D = decimal.Decimal
BASE = Path('/workspace/work/recover-small_h2_910/m16/approaches/budget_collapse_review')
TRUTH = Path('/workspace/clean-neurons/m16/approaches/ht_seeds/scan_seeds.json')
truth = json.loads(TRUTH.read_text(), parse_float=D)
saved = json.loads((BASE / 'independent_analysis.json').read_text())
rows, hashes = [], {}
for t in (4, 5, 6):
    path = BASE / f'independent_panel_T{t}.json'
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    panel = json.loads(path.read_text(), parse_float=D)
    for model in panel['models']:
        tr = truth[f"16:{t}:{model['seed']}"]['truth']
        p, se = tr['acc'], tr['se']
        for cell in model['budgets']:
            estimates = [r['estimate'] for r in cell['rules']]
            costs = [r['cold_fpe'] for r in cell['rules']]
            kinds = [[r['kind_rule_seeds'][k] for r in cell['rules']] for k in ('23', '12')]
            assert not set(kinds[0]) & set(kinds[1])
            assert all(len(set(a)) == 16 for a in kinds)
            mean = sum(estimates) / D(16)
            variance = sum((x - mean) ** 2 for x in estimates) / D(15)
            mse = sum((x - p) ** 2 for x in estimates) / D(16) - se**2
            cost = sum(costs) / D(16)
            assert mse > 0 and variance > 0
            gains = {key: p * (1-p) / cost / denominator
                     for key, denominator in [('mse', mse), ('variance', variance)]}
            original = next(r for r in saved['rows'] if (r['T'], r['seed'], r['budget']) ==
                            (t, model['seed'], cell['target_budget']))
            rows.append(dict(T=t, seed=model['seed'], budget=cell['target_budget'],
                             gains={key: float(value) for key, value in gains.items()},
                             max_relative_difference=max(abs(float(value) / original['metrics'][key]['gain'] - 1)
                                                         for key, value in gains.items()),
                             budget_ratio=float(cost / D(cell['target_budget']))))
geometric = []
for budget in (16000, 64000):
    for metric in ('mse', 'variance'):
        group = {}
        for t in (4, 5, 6):
            values = [D(str(r['gains'][metric])) for r in rows if r['budget'] == budget and r['T'] == t]
            group[t] = (sum(v.ln() for v in values) / D(5)).exp()
        geometric.append(dict(budget=budget, metric=metric,
                              gains={str(t): float(g) for t, g in group.items()},
                              declines={f'{a}->{b}': float(group[a]/group[b]) for a,b in ((4,5),(5,6))}))
out = dict(rows=rows, geometric=geometric, panel_hashes=hashes,
           maximum_relative_gain_difference=max(r['max_relative_difference'] for r in rows),
           cpu_seconds=time.process_time()-START,
           scope='Independent point arithmetic and seed namespace check; no bootstrap coverage certification.')
Path(__file__).with_name('budget_review_results.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('rows','panel_hashes')}, indent=2))
