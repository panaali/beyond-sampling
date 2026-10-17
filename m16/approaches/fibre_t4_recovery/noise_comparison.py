"""Single-case descriptive comparison retaining truth uncertainty and cost scope."""
import json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
est=json.loads((HERE/'noise_summary.json').read_text());ref=json.loads((HERE/'reference_summary.json').read_text());assert est['model_sha256']==ref['model_sha256']
e=est['estimate'];p=ref['accuracy'];lo,hi=ref['binomial_CP95'];assert 0<lo<hi<1 and 0<e<1
err=e-p;cost=est['legacy_ops'];interval=[e-hi,e-lo]
def ratio(q,unit):return cost*(e-q)**2/(unit*q*(1-q))
def budgets(unit):
 vals=[unit*q*(1-q)/(e-q)**2 for q in (lo,hi)]
 return {'minimum_over_CP95':min(vals),'maximum_over_CP95':None if lo<=e<=hi else max(vals),'unbounded_maximum':lo<=e<=hi}
out={'model_sha256':est['model_sha256'],'estimate':e,'reference':p,'reference_n':ref['n'],'reference_CP95':[lo,hi],'error':err,'error_interval_CP95':interval,'absolute_error_upper_CP95':max(abs(v) for v in interval),'legacy_ops_including_zero_guard':cost,'original_legacy_without_guard':cost-6*est['guard']['calls'],'reference_standard_error':ref['standard_error'],'historical_ratio_320':ratio(p,320),'historical_variance_subtracted_floor_ratio_320':max(err**2-ref['standard_error']**2,0)*cost/(320*p*(1-p)),'legacy_counter_ratio_CP95_320':[0 if lo<=e<=hi else min(ratio(lo,320),ratio(hi,320)),max(ratio(lo,320),ratio(hi,320))],'conditional_sampling_break_even_actual_cost_512':budgets(512),'scope':'One noisy model only. Legacy estimator counter is not complete actual cost. Conditional 95% cost budget uses separately counted512 sampler and binomial confidence set; no certified cost win, deterministic-error certificate, rung or group inference.'}
(HERE/'noise_comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
