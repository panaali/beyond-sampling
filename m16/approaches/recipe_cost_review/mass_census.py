"""Recipe mass-cost contraction with source-identical determinant guard only."""
import json,time
from collections import Counter
import numpy as np
from run import HERE,PRIOR,SOURCE,PINS
start=time.process_time()
for exponent,expected in [(-40,True),(-39,False)]:
    R=np.diag([2.**exponent,1.,1.,1.]);got=abs(np.linalg.det(R))<1e-12;assert got==expected
    assert (abs(np.linalg.det(R[[1,0,2,3]]))<1e-12)==expected
archive=np.load(SOURCE/'error_captured.npz');rays=archive['rays'];N=len(rays);assert N==28000
source=np.empty(N);doubled=np.empty(N);seen=[]
for batch in range(8):
    data=np.load(SOURCE/f'batch_{batch}.npz');ids=data['indices'];seen.extend(ids.tolist());source[ids]=data['source'];doubled[ids]=data['doubled']
assert seen==list(range(N))
determinants=np.array([float(np.linalg.det(R)) for R in rays]);assert np.isfinite(determinants).all()
cutoff=np.abs(determinants)<1e-12;assert np.all(source[cutoff]==0) and np.all(doubled[cutoff]==0)
summary=json.loads((SOURCE/'summary.json').read_text());remaining=summary['doubled_mass_named_ops']-156*N;assert remaining%152==0
nodes=remaining//152;assert nodes==2110488 and nodes%24==0
assert summary['original_mass_named_ops']==156*N+152*(nodes//2)
countcut=int(cutoff.sum());thin_numerator=nodes//24-(N-countcut);assert thin_numerator%3==0
thin=thin_numerator//3;wide=N-countcut-thin;assert min(thin,wide)>=0
counts=dict(wide=wide,thin=thin,cutoff=countcut)
branch={r['name']:r for r in json.loads((PRIOR/'controls.json').read_text())['rows']}
legacy=sum(counts[k]*r['legacy'] for k,r in branch.items());assert legacy==summary['doubled_mass_named_ops']
categories=Counter();opaque=Counter()
for name,row in branch.items():
    for key,value in row['ledger']['totals'].items():categories[key]+=counts[name]*value
    for item in row['ledger']['opaque_calls']:opaque[tuple(item['key'])]+=counts[name]*item['count']
visible=sum(categories.values());assert visible==200*N+192*nodes-16*countcut
np.savez_compressed(HERE/'det_census.npz',determinants=determinants,cutoff=cutoff)
result=dict(counts=counts,quadrature_nodes=nodes,legacy=legacy,visible=visible,categories=dict(categories),opaque=[dict(key=list(k),count=v) for k,v in sorted(opaque.items())],
 diagnostic_determinant_calls=N+4,diagnostic_covariance_inverse_calls=0,diagnostic_mass_calls=0,cpu_seconds=time.process_time()-start,
 pins={str(k):v for k,v in PINS.items()},complete_implementation_cost=False)
(HERE/'mass_census.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('pins','categories','opaque')}))
