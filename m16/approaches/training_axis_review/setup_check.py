"""Separate cold-process setup measurement; no statistical estimator result."""
import time,json
from pathlib import Path
start=time.process_time();wall=time.perf_counter()
import loaders as L
modules=time.process_time()-start
m=L.load(24,4);loaded=time.process_time()-start
# Universal geometry may be stored as literal advice; disclose cold current implementation cost.
t=time.process_time();halves=L.COUNT.R._halves();geometry=time.process_time()-t
out=dict(import_cpu=modules,import_and_first_torch_load_cpu=loaded,universal_r3_geometry_rebuild_cpu=geometry,halves=len(halves),total_cpu=time.process_time()-start,wall=time.perf_counter()-wall,model_dependent_geometry=False,geometry_operation_count='not automatically instrumented: numpy inverse/normalization use ordinary numbers; per-model panel counts are model-dependent circuit cost, not a complete cold executable tally',original_panel_import_overhead='per-cell timers exclude initial loader import; first-cell timers include first torch import/model load; retries documented separately')
Path(__file__).with_name('setup_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
