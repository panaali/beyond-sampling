"""Test the independently derived pure-resolution panel factor, no model calls."""
import json
from pathlib import Path
import numpy as np
from review import split_statistics,line_statistics
p=Path(__file__).parent;out=[];grid=np.linspace(.05,.95,257)
base=line_statistics((grid*grid)[None,:],grid)
for path in sorted((p/'data').glob('T*.npz')):
 d=np.load(path);entry={'case':path.stem}
 for key in ('panel_specs','uniform_specs'):
  panels=[[] for _ in range(32)];factors=np.zeros(32)
  for i,lo,hi in d[key]:
   i=int(i);x=np.linspace(lo,hi,257);panels[i].append((lo,hi,x*x));factors[i]+=((hi-lo)/.9)**5
  truth=float(np.sqrt(factors.mean()));stats=split_statistics(panels)
  observed=np.array([r['rms']/b['rms'] for r,b in zip(stats['levels'],base['levels'])]);error=float(np.max(abs(observed-truth)))
  assert error<1e-8
  entry[key]=dict(predicted_factor=truth,observed_factors=observed.tolist(),max_absolute_error=error)
 out.append(entry)
(p/'quadratic_control.json').write_text(json.dumps(out,indent=2)+'\n')
print('max factor error',max(x[k]['max_absolute_error'] for x in out for k in ('panel_specs','uniform_specs')))
