"""Fixed one-model uncertainty comparison after exhaustive named-cost census."""
import hashlib,json
from fractions import Fraction as F
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=json.loads((HERE/'summary.json').read_text());cold=json.loads((HERE/'cold.json').read_text());assembly=json.loads((HERE/'assembly.json').read_text());pin=json.loads((HERE/'reference_pin.json').read_text())
assert s['complete'] and s['indices']==list(range(48)) and cold['complete'] and assembly['complete']
p=Path(pin['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'];ref=json.loads(p.read_text());assert s['original_order_estimate'].hex()==assembly['estimate'].hex()==ref['estimate'].hex()
cold_visible=sum(r['ledger']['inclusive_explicit_total'] for r in cold['modules'].values());outer=assembly['ledger']['inclusive_explicit_total'];assert cold_visible==6930 and outer==50
cost=s['visible_after_diagnostic']+cold_visible+outer;budget=cost//512
a=F(ref['estimate']);p0=F(ref['reference']);qs=list(map(F,ref['reference_CP95']))
def ratio(q):return F(budget)*(a-q)**2/(q*(1-q))
result={'model_sha256':ref['model_sha256'],'component_visible':s['visible_after_diagnostic'],'cold_visible':cold_visible,'outer_visible':outer,'total_visible':cost,'sampler_units':512,'integer_sample_budget':budget,'estimate':float(a),'reference':float(p0),'reference_n':ref['reference_n'],'reference_CP95':list(map(float,qs)),'point_ratio':float(ratio(p0)),'CP95_ratio_bounds':[0.0 if qs[0]<=a<=qs[1] else float(min(map(ratio,qs))),float(max(map(ratio,qs)))],'remaining_opaque_cold':[x for r in cold['modules'].values() for x in r['ledger']['opaque_calls']],'model_dependent_opaque':[x for r in s['rows'] for x in r['opaque_calls']],'remaining_cost_to_conditional_95_budget':ref['conditional_sampling_break_even_actual_cost_512']['minimum_over_CP95']-cost,'scope':'One fixed noisy model; binomial truth uncertainty and declared source units only. Opaque native setup, I/O, memory/control/library internals excluded. No hardware ceiling, deterministic accuracy certificate, full noisy-panel result or ladder pass.'}
(HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if 'opaque' not in k}))
