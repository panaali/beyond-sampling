#!/usr/bin/env python3
"""T384 bounded saved-evidence arithmetic. No model imports, RNG or experiments.

Run from the repository root:
    python3 review/tutorial_factcheck/verify_saved.py
Writes checks.json beside this script. Git objects named in sources.json must exist.
"""
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

started = time.process_time()
here = Path(__file__).resolve().parent
manifest = json.loads((here / 'sources.json').read_text())
sources = {r['key']: r for r in manifest['repository_sources']}


def source(key):
    r = sources[key]
    b = subprocess.check_output(['git', 'show', r['commit'] + ':' + r['path']])
    assert hashlib.sha256(b).hexdigest() == r['sha256'], key
    return b.decode()


def group(key):
    return json.loads(source(key))['budgets']['0']['groups']['retrain']


h = group('hp4')
screen = group('hp4_screen')
stack = group('hp4_stacked')
assert abs(h['C'] / h['F'] - h['n_star']) < 1e-9
assert round(h['F'], 2) == 2.63 and round(h['lo'], 2) == 1.93
assert round(h['hi'], 2) == 3.39 and h['n'] == 100
assert round(screen['F'], 2) == 2.67 and round(stack['F'], 2) == 2.98
panels = json.loads(source('panels'))
acc = panels['dev3']['acc']
assert len(acc) == 100
rows = panels['t234']['rows']
assert [r['n'] for r in rows] == [100, 100, 100, 100, 25]
w = json.loads(source('weights'))['models']['M_16,10']
assert (w['H'], w['T']) == (16, 10)
def scalar_count(value):
    return sum(map(scalar_count, value)) if isinstance(value, list) else 1
assert sum(scalar_count(w[k]) for k in ['Wi', 'Wh', 'Wo']) == 432
page = source('baseline')
match = re.search(r'const L = (\[\[.*?\]\]);', page)
points = json.loads(match.group(1))
log = source('law')
logged = [(int(t), float(e), float(g)) for t, e, g in re.findall(
    r'T(\d+) s\d+: err ([\d.]+)\s+iid gain\s+([\d.]+)', log)]
assert [tuple(p) for p in points] == logged and len(points) == 30
by_t = {}
for t in sorted({p[0] for p in points}):
    a = [p[2] for p in points if p[0] == t]
    by_t[t] = {'n': len(a), 'geometric_mean_gain': math.exp(sum(map(math.log, a))/len(a)),
               'individual_gains_above_one': sum(x > 1 for x in a)}
p = .9531175
e32 = json.loads(source('e32_data'))
e32rows = [r for v in e32['seeds'].values() for r in v['rows']]
assert len(e32rows) == 1024 and e32['protocol']['ops_per_sample'] == 72
mean = lambda values: sum(values)/len(values)
pq32 = mean([r['truth']*(1-r['truth']) for r in e32rows])
a_mse = mean([(r['A']-r['truth'])**2 for r in e32rows])
b_mse = mean([(r['B']-r['truth'])**2 for r in e32rows])
exact_ops = mean([r['exact_ops'] for r in e32rows])
e32_check = {
    'n': len(e32rows), 'mean_pq': pq32,
    'mean_exact_ops': exact_ops, 'mean_leaf_count': mean([r['leaves'] for r in e32rows]),
    'A_ratio_ARC_71040_ops': a_mse*(71040/72)/pq32,
    'A_ratio_mean_measured_ops': a_mse*(mean([r['A_ops'] for r in e32rows])/72)/pq32,
    'B_ratio_ARC_7680_ops_uncharged_integral_sampling': b_mse*(7680/72)/pq32,
    'sampling_to_point001_RMS_cost_divided_by_mean_exact_cost': pq32/.001**2*72/exact_ops,
}
report = {
    'scope': 'saved public aggregates, embedded data and scalar arithmetic only',
    'hp4_official': h,
    'hp4_screen_sensitivity': {'C': screen['C'], 'F': screen['F']},
    'hp4_stacked_sensitivity': {'C': stack['C'], 'F': stack['F']},
    'sampling_rms_at_hp4_cost_pp': h['rms_pp'] / math.sqrt(h['F']),
    'target_300B_one_standard_error_pp': 100 * math.sqrt(p * (1-p) / 3e11),
    'target_samples_for_point001_rms': p * (1-p) / .001**2,
    'panel_variance_assumed_by_74000_guide': 74000 * .001**2,
    'noise_20000_SE_pp': {'target': 100*math.sqrt(p*(1-p)/20000),
                        'maximum': 100*math.sqrt(.25/20000)},
    'blind_mass': (1+10)/2**10,
    'all_negative_mass': 1/2**10,
    'all_negative_max_error_pp': 100/2**10,
    'dev3': {'n': len(acc), 'minimum': min(acc), 'maximum': max(acc),
             'mean': sum(acc)/len(acc), 'count_at_or_below_target': sum(x <= p for x in acc)},
    'training_rows': rows,
    'parameter_count': 432,
    'top3_placements': 10*9*8,
    'density_extreme_bin_ratio_original': 358060/2872,
    'figure5_saved_log_match': True,
    'figure5_by_T': by_t,
    'E32_saved_1024_row_check': e32_check,
    'gain_identity': 'For q=1-p, n faces, C=n*c: [p*q/C]/[q*q*CV^2/n] = p/(q*CV^2*c).',
    'sampling_F_below_floor': 'For C<1000, expected F=1000/C; for C>=1000, expected F=1.',
    'model_calls': 0,
    'random_draws': 0,
    'process_cpu_seconds': time.process_time()-started,
}
(here/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
print('PASS: saved HP4 aggregates, 100 DEV3 values, five training rows, 30 plotted points and scalar identities.')
