"""Oracle overlap correction for frozen Gaussian marginals; zero new model calls."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def group(name):
    return 'orig' if name == 'orig' else 'seed' if name.startswith('seed') else name.rsplit('_', 1)[0]


def evaluate(rows):
    out = []
    for budget in sorted({r['budget'] for r in rows}):
        for g in sorted({group(r['model']) for r in rows}):
            rs = [r for r in rows if r['budget'] == budget and group(r['model']) == g]
            assert rs
            base = sum(r['truth']*(1-r['truth'])/r['cost'] for r in rs)/len(rs)
            result = dict(budget=budget, group=g, n=len(rs), actual_cost=rs[0]['cost'])
            for z in (0, 5):
                errors = []
                for r in rs:
                    low, high = r['est'], r['est']+r['union_gap']
                    assert -1e-12 <= low <= high <= 1+1e-12
                    p, se = r['truth'], r['truth_se']
                    distance = max(low-(p+z*se), (p-z*se)-high, 0)
                    errors.append(distance**2-se**2)
                result[f'oracle_ratio_truth_shift_{z}se'] = sum(errors)/len(errors)/base
            out.append(result)
    return out


def main():
    raw = json.loads((HERE.parent/'candidate_raw.json').read_text())
    output = dict(interpretation='Optimistic oracle with frozen Gaussian marginals, free arbitrary dependence correction. Not an estimator or statistical confidence bound.',
                  model_evaluations=0, results=evaluate(raw['rows']))
    (HERE/'results.json').write_text(json.dumps(output, indent=2)+'\n')
    for r in output['results']:
        print(r['budget'],r['group'],r['oracle_ratio_truth_shift_0se'],r['oracle_ratio_truth_shift_5se'])


if __name__ == '__main__':
    main()
