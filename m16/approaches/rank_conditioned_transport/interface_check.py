import json
from pathlib import Path
import numpy as np
from estimator import est

model = (np.array([.7, -.4]), np.array([[.8, -.3], [.2, .9]]), np.zeros((3, 2)))
value, cost = est(model, 1024, np.random.default_rng(0))
assert abs(value-1/3) < 1e-12
assert np.isfinite(cost) and cost > 0
row = dict(estimate=value, exact=1/3, error=value-1/3, requested_advisory_budget=1024,
           measured_forward_equivalents=cost, hardware_dependent_cost=True,
           calibration_included=True, total_error_certified=False)
Path(__file__).with_name('interface_results.json').write_text(json.dumps(row, indent=2)+'\n')
print(json.dumps(row))
