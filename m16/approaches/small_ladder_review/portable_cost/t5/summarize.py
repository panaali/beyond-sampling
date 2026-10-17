"""Combine saved curves and frozen reference intervals; no new evaluations."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'results.json').read_text())['models'][:2]+json.loads((HERE/'corrected_sigma002.json').read_text())['models']
out=dict(cases=[],qualification='Every interval is integration-only and conditional on its floating geometry; no total-error certificate.')
for r in rows:
 curves=[]
 for c in r['curves']:
  if c['order'] not in (24,32):continue
  lo=max(0.,r['reference']-r['reference_bound']);hi=min(1.,r['reference']+r['reference_bound'])
  variance=min(lo*(1-lo),hi*(1-hi))/c['affine_only_forward_equivalents']
  error=abs(c['difference_from_reference'])+r['reference_bound']
  curves.append(dict(order=c['order'],accuracy=c['accuracy'],full_scalar_operation_upper_bound=c['cost']['known_arithmetic_upper_bound'],affine_only_forward_equivalents=c['affine_only_forward_equivalents'],integration_only_bound=c['integration_only_bound'],own_bound_squared_over_sampling_variance=c['integration_bound_squared_over_sampling_variance'],reference_interval_error_upper_bound=error,minimum_sampling_variance_over_reference_interval=variance,reference_interval_error_squared_over_sampling_variance=error*error/variance,end_to_end_cpu=c['end_to_end_cpu']))
 out['cases'].append(dict(name=r['name'],model_sha256=r['model_sha256'],curves=curves))
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
