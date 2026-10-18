"""Compose recipe traces and isolate the routines absent from exact mode."""
from collections import Counter
import hashlib,json
from run import HERE,PRIOR,SOURCE
rows=[json.loads((HERE/f'geometry_{b}.json').read_text()) for b in range(8)]
sha=hashlib.sha256((HERE/'run.py').read_bytes()).hexdigest();assert all(r['adapter_sha256']==sha for r in rows)
assert [(r['first'],r['last']) for r in rows]==[(12*b,12*(b+1)) for b in range(8)]
assert sum(r['legacy_geometry'] for r in rows)==36398526
assert sum(r['stats']['cells'] for r in rows)==6182
assert sum(r['mass_callbacks'] for r in rows)==28000
assert all(r['cold_setup']==rows[0]['cold_setup'] and r['rule_setup']==rows[0]['rule_setup'] for r in rows)
categories=Counter();opaque=Counter();functions=Counter();native=Counter()
for r in rows:
    categories.update(r['native']['totals'])
    for site,items in r['native']['sites'].items():functions[site.split(':')[0]]+=sum(items.values())
    returns=[items for site,items in r['native']['sites'].items() if site.startswith('estT:') and site.endswith('1.0 - err')]
    assert returns==[{'scalar_Sub':1}]
    for field,out in [('opaque_calls',opaque),('native_calls',native)]:
        for item in r['native'][field]:out[tuple(item['key'])]+=item['count']
categories['scalar_Sub']-=7;functions['estT']-=7;geometry=sum(categories.values());assert geometry==sum(functions.values())
mass=json.loads((HERE/'mass_census.json').read_text());setup=rows[0]['cold_setup'];rules=rows[0]['rule_setup']
for r,field in [(mass,'opaque'),(setup,'opaque_calls'),(rules,'opaque_calls')]:
    for item in r[field]:opaque[tuple(item['key'])]+=item['count']
visible=geometry+mass['visible']+setup['inclusive_explicit_total']+rules['inclusive_explicit_total']
exact=json.loads((PRIOR/'summary.json').read_text());saved=json.loads((SOURCE/'summary.json').read_text());p=saved['accuracies']['positive_exact'];candidate=saved['accuracies']['doubled'];error=abs(candidate-p)
# Reference uncertainty is the exact reference's conditional integration bound,
# since the actual candidate value is compared directly with positive_exact.
exact_reference=json.loads((__import__('inventory').AUTHOR/'plackett_rule_doubling/summary.json').read_text());bound=exact_reference['conditional_reference_integration_bound']
low=512*(p+bound)*(1-p-bound)/(error+bound)**2;high=512*(p-bound)*(1-p+bound)/(error-bound)**2
result=dict(ordered_rays_verified=28000,terminal_cells=6182,legacy_geometry=36398526,geometry_explicit=geometry,geometry_categories=dict(categories),geometry_function_totals=dict(functions),
 recipe_only_weights_to_margin=functions['_weights_to_margin'],recipe_only_frozen_margin=functions['_frozen_margin'],
 mass_explicit=mass['visible'],cold_halves_explicit=setup['inclusive_explicit_total'],cold_rules_explicit=rules['inclusive_explicit_total'],
 total_visible_explicit_with_cold_setup=visible,visible_vs_exact=visible/exact['total_visible_explicit_with_cold_setup'],
 opaque_calls=[dict(key=list(k),count=v) for k,v in sorted(opaque.items())],native_calls=[dict(key=list(k),count=v) for k,v in sorted(native.items())],
 sampler_explicit=512,candidate_accuracy=candidate,reference_accuracy=p,conditional_reference_integration_bound=bound,absolute_bias=error,
 conditional_single_case_break_even_total_units_interval=[low,high],visible_only_squared_bias_ratio_interval=[visible/high,visible/low],
 trace_function_cpu_sum=sum(r['cpu_seconds'] for r in rows),max_rss_kib=max(r['max_rss_kib'] for r in rows),complete_implementation_cost=False,
 scope='Visible source expressions only. Rank/inverse/det/native internals, membership/control/memory remain unpriced. Conditional reference assumes captured geometry/roundoff. No full-cost, sampling-win, noise-panel or rung claim; no choice between recipes based on this single checkpoint.')
(HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('geometry_categories','geometry_function_totals','opaque_calls','native_calls')}))
