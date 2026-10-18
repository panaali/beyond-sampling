"""Independent scalar-power attribution on the seven retained failures."""
import sys,json,time,math,hashlib
from pathlib import Path
import numpy as np
here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h4_4/m16/approaches/vector_triangle_selection');sys.path.insert(0,str(src));from radial_probe import cases,scalar_radial
from pow_selector import hybrid_radial
start=time.process_time();allv=cases();saved=json.loads((src/'attribution.json').read_text());rows=[]
for i in saved['pure_vector_mismatch_indices']:
 v=allv[i];s=v.sum(axis=0);c=s/math.sqrt(float(s@s));p=float(np.min(v@c));scalar=p**2;array=float((np.array([p])**2)[0]);assert scalar!=array
 actual=hybrid_radial(v[None])[0];expected=np.array(scalar_radial(v));assert np.array_equal(actual,expected)
 rows.append(dict(index=i,projection=p,python_pow=scalar,numpy_square=array,difference=scalar-array))
assert len(rows)==7
import statistics
pilot=json.loads((src/'pow_pilot.json').read_text());base=pilot['baseline'];candidate=pilot['candidate'] if 'candidate' in pilot else pilot['guarded'];assert base['rows']==candidate['rows'] and len(base['rows'])==64
medians={k:statistics.median(r['cpu'] for r in pilot['timings'] if r['variant']==k) for k in ('grouped','guarded')}
assert medians==pilot['median_cpu'] and medians['guarded']/medians['grouped']==pilot['guarded_over_grouped']
for record in (base,candidate):assert sum(record['cost']['parts'].values())==record['cost']['known_arithmetic_upper_bound']
out=dict(pilot_medians=medians,pilot_ratio=pilot['guarded_over_grouped'],baseline_named_allowance=base['cost']['known_arithmetic_upper_bound'],candidate_named_allowance=candidate['cost']['known_arithmetic_upper_bound'],cases=rows,review_cpu=time.process_time()-start,scope='Seven preserved failures independently reproduced; finite panel attribution, not platform-independent BLAS/roundoff equivalence.')
(here/'radial_pow_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
