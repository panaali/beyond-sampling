"""Independent exact dyadic pair signs and saved positive mass sums."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
import json,time
import numpy as np
start=time.process_time();p=Path('/workspace/work/recover-small_h2_34/m16/approaches');d=json.loads((p/'acute_t4_reference/results.json').read_text());R=np.load(p/'class_region_t4/quadratic_exact_mass_calls.npz')['rays'];v=np.load(p/'acute_t4_reference/acute_values.npz');source=np.load(p/'cone_mass4_review/quadratic_values.npz')['values'];acute=[];nonacute=[]
for row in d['rows']:
 i=row['index'];r=[[F(float(x)) for x in y] for y in R[i]];dots=[sum(a*b for a,b in zip(r[j],r[k])) for j,k in combinations(range(4),2)];signs=[int(x>0)-int(x<0) for x in dots];assert signs==row['exact_pair_signs']
 if min(signs)<0:nonacute.append(i);assert row['rotation_obstruction'] and not np.isfinite(v['values'][i])
 else:acute.append(i);assert row['value']==v['values'][i] and row['value']>=0
assert len(acute)==432 and nonacute==d['uncovered'] and len(nonacute)==56
sumf=lambda a:sum((F(float(x)) for x in a),F(0));mass=sumf(v['values'][acute]);bound=sumf(v['bounds'][acute]);difference=sumf(source[acute])-mass
assert abs(float(mass)-d['positive_mass_sum'])<1e-17 and abs(float(bound)-d['integration_only_bound_sum'])<1e-20
out={'exact_acute_cones':len(acute),'exact_negative_pair_obstructions':len(nonacute),'positive_mass':float(mass),'integration_only_bound':float(bound),'source_minus_positive':float(difference),'cpu_seconds':time.process_time()-start,'scope':'stored ray pair signs and raw sums; no new quadrature or floating certificate'}
Path(__file__).with_name('acute_reference_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
