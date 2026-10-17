"""Independent exact stored geometry and Decimal mass aggregation."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D,getcontext
from itertools import permutations
from collections import Counter
import gzip,hashlib,json,math,time
import numpy as np
start=time.process_time();getcontext().prec=60;base=Path('/workspace/work');p=base/'recover-small_h2_34/m16/approaches';inventory=json.loads((p/'positive_t4_corpus/inventory.json').read_text());capture=p/'class_region_t4/quadratic_exact_mass_calls.npz'
assert hashlib.sha256(capture.read_bytes()).hexdigest()==inventory['capture_sha256'];rays=np.load(capture)['rays'];counts=Counter()
for row in inventory['rows']:
 R=[[F(float(v)) for v in ray] for ray in rays[row['index']]];counts[row['mode']]+=1
 if row['mode']=='uncovered':continue
 if row['mode']=='rotated':
  c,s=F(row['c']),F(row['s']);assert c*c+s*s==1;i,j=row['pair']
  for r in R:a,b=r[i],r[j];r[i]=c*a-s*b;r[j]=s*a+c*b
 assert all(all(r[q]>=0 for r in R) or all(r[q]<=0 for r in R) for q in range(4))
assert dict(counts)==inventory['counts']
p=base/'recover-small_h2_910/m16/approaches';d=json.loads((p/'bulk_rank_mass/results.json').read_text());art=Path('/workspace/clean-neurons/m16/approaches/small_ladder_review/rational_geometry/geometry_h4_t4.json.gz');geometry=json.load(gzip.open(art,'rt'))
def det(A):
 n=len(A);return sum((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))*math.prod(A[i][p[i]] for i in range(n)) for p in permutations(range(n)))
parents=0
for row in d['parents']:
 s=geometry['gate_simplices'][row['index']];R=[list(map(int,r)) for r in s['integer_rays']]
 expected=F(abs(det(R)),math.prod(sum(map(abs,r)) for r in R));assert expected==F(*map(int,row['normalized_det']))
 for i in range(4):
  rows=[r for j,r in enumerate(R) if j!=i];n=[(-1)**q*det([r[:q]+r[q+1:] for r in rows]) for q in range(4)];values=[sum(a*b for a,b in zip(n,r)) for r in R]
  assert values[i]!=0 and all(v==0 for j,v in enumerate(values) if j!=i)
 assert all(all(r[q]>=0 for r in R) or all(r[q]<=0 for r in R) for q in range(4));assert row['partition_exact'];parents+=1
sums={}
for classification in d['totals']:
 selected=[r for r in d['pairs'] if r['classification']==classification]
 sums[classification]={k:sum((D(str(r[k])) for r in selected),D(0)) for k in ('mass20','bound20','correct_mass20')}
 for k,v in sums[classification].items():assert abs(v-D(str(d['totals'][classification][k])))<D('2e-16')
covered=sums['uniformly_correct']['mass20']+sums['uniformly_wrong']['mass20'];unresolved=D(1)-covered
assert abs(unresolved-D(str(d['unresolved_mass20'])))<D('2e-16')
out={'exact_rotation_inventory':dict(counts),'exact_parent_cofactors_and_determinants':parents,'covered_uniform_mass':str(covered),'unresolved_mass':str(unresolved),'uniform_integration_bound':str(sums['uniformly_correct']['bound20']+sums['uniformly_wrong']['bound20']),'cpu_seconds':time.process_time()-start,'scope':'geometry/source and raw sums independently checked; no new integration or independent reclipping of children'}
Path(__file__).with_name('rank_mass_rotation_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
