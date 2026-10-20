"""Independent T461 saved-record check; python review/dri_takeover_readiness/check_t461_decimal.py.
Uses Git text blobs and Decimal only; imports no author/model code.
"""
import decimal
import json
import subprocess

D = decimal.Decimal
decimal.getcontext().prec = 60
PIN = 'a52da0d6a57d29aa2f76daa74c3360222f33aec2'


def read(name):
    data = subprocess.check_output(['git', 'show', PIN + ':m16/approaches/is_v4/' + name])
    return json.loads(data, parse_float=D)


rows = []
for i in range(15):
    est, truth = read(f'est_{i}.json'), read(f'truth_{i}.json')
    delta = est['p'] - truth['truth']
    z = delta / (est['se'] ** 2 + truth['se'] ** 2).sqrt()
    before = {tuple(v) for v in est['triples_before']}
    after = {tuple(v) for v in est['triples_after']}
    assert len(before) == len(est['triples_before'])
    assert len(after) == len(est['triples_after'])
    rows.append(dict(error2=delta ** 2, pq=truth['truth'] * (1 - truth['truth']),
                     C=est['C'], Cd=est['C_draws'], z=z, ratio=est['p'] / truth['truth'],
                     truth_variance=truth['se'] ** 2, grew=len(after) > len(before),
                     new=bool(after - before), added=len(after-before), removed=len(before-after)))
S = lambda key: sum(r[key] for r in rows)
result = dict(F=S('error2') * S('C') / (15 * S('pq')),
              F_draws=S('error2') * S('Cd') / (15 * S('pq')),
              z_pass=sum(abs(r['z']) <= 2 for r in rows), max_abs_z=max(abs(r['z']) for r in rows),
              min_ratio=min(r['ratio'] for r in rows), max_ratio=max(r['ratio'] for r in rows),
              truth_variance_over_observed_mse=S('truth_variance') / S('error2'),
              cardinality_grew=S('grew'), new_triples_present=S('new'),
              s503_added=rows[3]['added'], s503_removed=rows[3]['removed'])
assert result['z_pass'] == 15 and result['cardinality_grew'] == 14 and result['new_triples_present'] == 15
print(json.dumps(result, indent=2, default=str))
