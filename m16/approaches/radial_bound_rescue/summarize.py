"""Merge only the32 frozen rescues; preserve all6315 prior values bitwise."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h2_34/m16/approaches')
parts=[json.loads((HERE/f'chunk_{i}.json').read_text()) for i in range(4)]
assert all(p['complete'] for p in parts)
rows=sum([p['rows'] for p in parts],[])
old=json.loads((BASE/'analytic_inner_mass/aggregate.json').read_text())
assert [r['index'] for r in rows]==old['unresolved_indices'] and len(rows)==32
assert all(r['mode']=='returned' and r['comparison_pass'] for r in rows)
saved=np.load(BASE/'analytic_inner_mass/aggregate_values.npz')
values=saved['values'].copy();bounds=saved['bounds']+saved['series_bounds'];previous=np.isfinite(values)
assert np.flatnonzero(~previous).tolist()==old['unresolved_indices']
for r in rows:values[r['index']]=r['value'];bounds[r['index']]=r['conditional_bound']
assert np.array_equal(values[previous],saved['values'][previous])
reference=np.load(BASE/'radial_t4_reference/combined_values.npz')
kernel=Counter();rational=Counter();cold=Counter()
for r in rows:
    kernel.update(r['kernel_counts']);rational.update(r['exact_operation_inventory']['operations'])
for p in parts:cold.update(p['cold_rule_counts'])
out=dict(complete=True,coverage=len(values),rescued=32,unchanged_prior_values=int(previous.sum()),
         error_probability=math.fsum(values),reference_error_probability=math.fsum(reference['values']),
         error_difference=math.fsum(float(x-y) for x,y in zip(values,reference['values'])),
         accuracy=1-math.fsum(values),reference_accuracy=1-math.fsum(reference['values']),
         accuracy_difference=-math.fsum(float(x-y) for x,y in zip(values,reference['values'])),
         array_semantics='values contains error-cone probability masses; accuracy is one minus their sum.',
         reporting_correction='Initial summary mislabeled error probability as accuracy; initial JSON and script retained. Numerical arrays unchanged.',
         conditional_bound_sum=math.fsum(bounds),reference_conditional_bound_sum=math.fsum(reference['bounds']),
         new_bound_sum=math.fsum(r['conditional_bound'] for r in rows),maximum_new_absolute_error=max(abs(r['error']) for r in rows),
         new_orders=dict(Counter(r['order'] for r in rows)),stage_paths=dict(Counter('/'.join(s['method'] for s in r['stages']) for r in rows)),
         minimum_lower_bound_improvement=min(float(F(r['stages'][-1]['certificate']['a']))/r['previous_a'] for r in rows),
         maximum_lower_bound_improvement=max(float(F(r['stages'][-1]['certificate']['a']))/r['previous_a'] for r in rows),
         new_kernel_counts=dict(kernel),new_kernel_named_total=sum(kernel.values()),
         prior_named_total=old['named_total_including_comparisons'],
         known_prior_plus_new_kernel_named_total=old['named_total_including_comparisons']+sum(kernel.values()),
         diagnostic_four_cold_rule_counts=dict(cold),exact_rational_operation_counts=dict(rational),
         max_rational_numerator_bits=max(r['exact_operation_inventory']['max_numerator_bits'] for r in rows),
         max_rational_denominator_bits=max(r['exact_operation_inventory']['max_denominator_bits'] for r in rows),
         exact_certificate_cpu=sum(s['certificate_cpu'] for r in rows for s in r['stages']),
         high_precision_selection_cpu=sum(s.get('selection_cpu',0) for r in rows for s in r['stages']),
         kernel_cpu=sum(r['kernel_cpu'] for r in rows),
         trained_cpu=sum(p['cpu_seconds_including_imports'] for p in parts),
         controls_cpu=json.loads((HERE/'controls.json').read_text())['cpu_seconds_including_imports'],
         maximum_80_to_100_bound_difference=max(abs(r['bound_80_to_100_difference']) for r in rows),
         complete_portable_cost=False,full_floating_certificate=False,
         cost_qualification='Known float count excludes rational/gcd/MP selection internals and memory/library costs; four cold plans are diagnostic replay overhead, separate from retained baseline cache.',
         bound_qualification='Inherited6315 bound assumptions and all general floating geometry/kernel/rule errors remain; determinant factor discrepancy is explicitly added only for32 new cones.')
np.savez_compressed(HERE/'combined_values.npz',values=values,conditional_bounds=bounds)
out['combined_archive_sha256']=hashlib.sha256((HERE/'combined_values.npz').read_bytes()).hexdigest()
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
