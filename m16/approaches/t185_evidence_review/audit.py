"""Saved-data contraction only. Never import the estimator or generate inputs."""
import hashlib
import json
import math
import time
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
from scipy.special import betainc, betaincinv

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results.json'
assert not OUT.exists(), 'Preserve prior evidence'
t0 = time.process_time()
manifest = json.loads((HERE / 'manifest.json').read_text())
for row in manifest['files']:
    b = (HERE / 'snapshot' / row['path']).read_bytes()
    assert len(b) == row['bytes'] and hashlib.sha256(b).hexdigest() == row['sha256']

def read(name):
    return json.loads((HERE / 'snapshot/m16/approaches/estT_fix' / name).read_text(), parse_float=Decimal)

def rational(x):
    q = F(x)
    return {'fraction': str(q), 'float': float(q)}

def ratio(a, p, n):
    return n * (a-p)**2 / (p*(1-p))

def interval_ratio(a, lo, hi, n):
    assert 0 < lo <= hi < 1
    ends = [ratio(a, lo, n), ratio(a, hi, n)]
    return [0 if lo <= a <= hi else min(ends), max(ends)]

assert interval_ratio(F(1, 2), F(1, 4), F(3, 4), 1) == [0, F(1, 3)]
assert interval_ratio(F(1, 10), F(1, 4), F(3, 4), 1) == [F(3, 25), F(169, 75)]

exact, recipe = read('run5/zoo0_exact.json'), read('run5/zoo0_recipe108.json')
p, a = F(exact['acc']), F(recipe['acc'])
sample_cost = F(6000)*800/5760
assert sample_cost == F(2500, 3)
n = F(recipe['ops']) / sample_cost
mc = F(read('mc_H8_T5.json')['zoo0'])
N = 10_000_000
hits = mc*N
assert hits.denominator == 1
k = int(hits)
lo = float(betaincinv(k, N-k+1, .025))
hi = float(betaincinv(k+1, N-k, .975))
se = math.sqrt(float(mc*(1-mc)/N))
nf, af = float(n), float(a)
# (n+1)p^2-(2na+1)p+na^2 <= 0.
disc = math.sqrt(1+4*nf*af*(1-af))
accept = [(2*nf*af+1-disc)/(2*(nf+1)), (2*nf*af+1+disc)/(2*(nf+1))]
for endpoint in accept:
    assert abs(ratio(af, endpoint, nf)-1) < 1e-9

v = read('val5.json')
rows = [[F(x) for x in r] for r in v['thin_rows(det,pl256,d128,d64,th4-10,th6-10,th10-pl)']]
assert len(rows) == 60 and all(len(r) == 7 for r in rows)
cols = {'log128': 2, 'log64': 3, 'thin4': 4, 'thin6': 5, 'thin10_vs_plackett': 6}
relative = {name: max(abs(r[c]/r[1]) for r in rows) for name, c in cols.items()}
for name, val in relative.items():
    assert math.isclose(float(val), float(v['max_rel'][name]), rel_tol=5e-15)
# Values derived from the saved differences, not recomputed masses.
series = {'pl256': [r[1] for r in rows],
          'pl128': [r[1]+r[2] for r in rows],
          'pl64': [r[1]+r[3] for r in rows],
          'thin10': [r[1]+r[6] for r in rows],
          'thin6': [r[1]+r[6]+r[5] for r in rows]}
thin_actual = [r for r in rows if r[0] < F(1, 1_000_000)]
t4 = read('rescore/zoo0.json')['recipe108']
result = {
 'scope': 'Saved one-clean-model audit; numerical CP conditional on source-documented MC provenance; abstract legacy cost only',
 'manifest_files_checked': len(manifest['files']),
 'one_model': {
   'estimate': rational(a), 'numerical_reference': rational(p), 'difference': rational(a-p),
   'difference_percentage_points': rational(100*(a-p)), 'legacy_sample_cost': rational(sample_cost),
   'legacy_fractional_draws': rational(n), 'floor_draws': n.numerator//n.denominator,
   'nominal_ratio': rational(ratio(a,p,n)),
   'floor_ratio': rational(ratio(a,p,n.numerator//n.denominator)),
   'recipe_to_exact_proxy_cost': rational(F(recipe['ops'],exact['ops'])),
   'recipe_proxy_per_cell': rational(F(recipe['ops'],recipe['cells'])),
   'timing_ratio_with_leaf_diagnostic': rational(F(recipe['sec'])/F(exact['sec'])),
   'acceptable_true_probability_interval': accept,
   'numerical_reference_location_in_acceptance_interval': min(float(p)-accept[0],accept[1]-float(p)),
 },
 'mc': {
   'source_documented_N': N, 'reconstructed_hits_not_saved_ledger': k, 'value': rational(mc),
   'se': se, 'reference_z': float(p-mc)/se, 'recipe_z': float(a-mc)/se,
   'cp95_numeric': [lo,hi],
   'beta_tail_residuals': [float(betainc(k,N-k+1,lo))-.025,float(betainc(k+1,N-k,hi))-.975],
   'raw_mc_point_ratio': rational(ratio(a,mc,n)),
   'ratio_interval': interval_ratio(af,lo,hi,nf),
 },
 'validation': {
   'rows':len(rows), 'actual_thin_branch_rows':len(thin_actual),
   'det_range':[float(min(r[0] for r in rows)),float(max(r[0] for r in rows))],
   'max_abs_columns': {name: float(max(abs(r[c]) for r in rows)) for name,c in cols.items()},
   'max_relative_columns': {name: float(q) for name,q in relative.items()},
   'max_actual_thin_vs_pl256_abs':float(max(abs(r[5]+r[6]) for r in thin_actual)),
   'derived_series_minima': {name:float(min(vals)) for name,vals in series.items()},
   'derived_series_negative_counts':{name:sum(q<0 for q in vals) for name,vals in series.items()},
   'halves_sum_minus_1':float(v['halves_sum_minus_1']),
   'simplex_z': [float(z[0]) for z in v['simplex_vs_mc_z']],
   'half_z_mixed_independent_numerator_and_se': [float(z) for z in v['half_vs_mc_z']],
 },
 'growth': {
   'same_zoo0_T4_to_T5_cells': rational(F(recipe['cells'],t4['cells'])),
   'same_zoo0_T4_to_T5_proxy_ops': rational(F(recipe['ops'],t4['ops'])),
   'assumed_13_5_growth_5_steps': rational(F(27,2)**5),
   'assumption_based_T10_proxy_cost': rational(recipe['ops']*F(27,2)**5),
   'monotone_countersequence_after_T5_cost': recipe['ops'],
   'interpretation': 'Constant continuation matches all finite observations; disproves inference from observations alone, not any independently proved algorithm bound',
 },
 'audit_cpu_seconds_excludes_imports_serialization':time.process_time()-t0,
 'model_or_mass_calls':0, 'MC_draws':0,
}
with OUT.open('x') as f:
    json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({key:result[key] for key in ['one_model','mc','validation','growth']},indent=2))
