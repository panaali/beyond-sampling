"""Compose the eight unchanged-geometry traces, retaining every opaque category."""
from collections import Counter
import json,hashlib
from inventory import HERE,AUTHOR
provenance=json.loads((HERE/'trace_provenance.json').read_text())
for name,sha in provenance['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
rows=[json.loads((HERE/f'geometry_{b}.json').read_text()) for b in range(8)]
assert [(r['first'],r['last']) for r in rows]==[(12*b,12*(b+1)) for b in range(8)]
assert sum(r['legacy_geometry'] for r in rows)==35650854
assert sum(r['stats']['cells'] for r in rows)==8455
assert sum(r['mass_callbacks'] for r in rows)==35098
assert all(r['cold_setup']==rows[0]['cold_setup'] and r['rule_setup']==rows[0]['rule_setup'] for r in rows)
categories=Counter();opaque=Counter();memory=Counter();native=Counter()
for r in rows:
    categories.update(r['native']['totals'])
    for name,out in [('opaque_calls',opaque),('memory_calls',memory),('native_calls',native)]:
        for item in r['native'][name]:out[tuple(item['key'])]+=item['count']
# Splitting one estT call into eight retains all parent-local expressions;
# only the final 1.0-err arithmetic is repeated seven extra times.
return_site='estT:L408:1.0 - err'
for r in rows:
    found=[(site,value) for site,value in r['native']['sites'].items() if site.startswith('estT:') and site.endswith('1.0 - err')]
    assert len(found)==1 and found[0][1]=={'scalar_Sub':1},found
categories['scalar_Sub']-=7
geometry=sum(categories.values());mass=json.loads((HERE/'mass_summary.json').read_text())
for item in mass['opaque']:opaque[tuple(item['key'])]+=item['count']
setup=rows[0]['cold_setup'];rules=rows[0]['rule_setup']
for r in (setup,rules):
    for item in r['opaque_calls']:opaque[tuple(item['key'])]+=item['count']
visible=geometry+mass['inclusive_explicit']+setup['inclusive_explicit_total']+rules['inclusive_explicit_total']
summary=json.loads((AUTHOR/'plackett_rule_doubling/summary.json').read_text());p=summary['reference_accuracy'];bound=summary['conditional_reference_integration_bound'];err=abs(summary['candidate_accuracy']-p)
# This is algebraic single-case headroom only, conditional on stored geometry/reference.
variance_low=(p+bound)*(1-p-bound);variance_high=(p-bound)*(1-p+bound)
budget_low=512*variance_low/(err+bound)**2;budget_high=512*variance_high/(err-bound)**2
result=dict(complete_trace=True,ordered_rays_verified=35098,terminal_cells=8455,legacy_geometry=35650854,
 geometry_explicit_after_removing_seven_extra_return_subtractions=geometry,geometry_categories=dict(categories),mass_explicit=mass['inclusive_explicit'],
 cold_halves_explicit=setup['inclusive_explicit_total'],cold_original_plus_candidate_rule_explicit=rules['inclusive_explicit_total'],
 total_visible_explicit_with_cold_setup=visible,opaque_calls=[dict(key=list(k),count=v) for k,v in sorted(opaque.items())],
 unpriced_geometry_memory_calls=[dict(key=list(k),count=v) for k,v in sorted(memory.items())],
 geometry_native_calls=[dict(key=list(k),count=v) for k,v in sorted(native.items())],
 sampler_explicit=512,conditional_single_case_break_even_total_units_interval=[budget_low,budget_high],
 conditional_remaining_allowance_after_visible_lower=budget_low-visible,
 visible_only_squared_bias_ratio_interval=[visible/budget_high,visible/budget_low],
 trace_function_cpu_sum=sum(r['cpu_seconds'] for r in rows),max_rss_kib=max(r['max_rss_kib'] for r in rows),
 complete_implementation_cost=False,scope='No portable full-cost or sampling-win claim. Opaque kernels, membership/control/memory, internal native implementations, geometry and reference arithmetic remain qualified. Function CPU excludes initial module imports and final JSON serialization; repeated diagnostic launch/setup is not charged eight times to one cold estimator.')
(HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('geometry_categories','opaque_calls','unpriced_geometry_memory_calls','geometry_native_calls')}))
