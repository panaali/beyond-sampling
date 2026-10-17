"""Same12 cone controls plus exact256-monomial nearconstant variance probes."""
import time
START=time.process_time()
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import itertools,json,math,resource
import numpy as np
import variant
P=Path(__file__).resolve().parent

def exact_moments(R):
 G=[[sum(F(x)*F(y) for x,y in zip(a,b)) for b in R] for a in R]
 mean=sum(G[i][j]*(2 if i==j else 1) for i in range(4) for j in range(4))/20
 second=F(0)
 for i,j,k,l in itertools.product(range(4),repeat=4):
  multiplicity=Counter((i,j,k,l));factor=math.prod(math.factorial(v) for v in multiplicity.values());second+=G[i][j]*G[k][l]*F(factor,840)
 return mean,second-mean*mean

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));M,old,kernel,rescue,radial,moment,source=variant.load();rules=kernel.Rules();out=dict(rows=[],probes=[],complete=False,selector_source=source)
 cases=[('orthant',np.eye(4),1/16),('diagonal',np.diag([1.,2.,3.,4.]),1/16),('nonacute',np.array([[1.,0,0,0],[-2,1,0,0],[0,0,1,0],[0,0,0,1]]),math.atan2(1,-2)/(8*math.pi)),('boundary',np.array([[1.,0,0,0],[2,1,0,0],[2,0,1,0],[2,0,0,1]]),None)]
 for name,R,truth in cases:
  for power in (-5,0,5):
   row=variant.one(R*2.**power,M,old,kernel,rescue,radial,rules);row.update(name=name,power=power,truth=truth)
   if row['mode']=='returned':
    assert row['bound']<=M.TOL
    if truth is not None:row.update(error=row['value']-truth,analytic_pass=abs(row['value']-truth)<=row['bound']+5e-14);assert row['analytic_pass']
   assert row['mode'] in ('returned','unresolved');out['rows'].append(row)
 for power in (-4,-12,-20,-28):
  R=np.array([[1.,0,0,0],[1,2.**power,0,0],[1,0,2.**power,0],[1,0,0,2.**power]]);op=old.Counter();a,b,det,data=moment.full_geometry(R,op);mu,var,second=moment.moments(data);exact_mu,exact_var=exact_moments(R)
  out['probes'].append(dict(power=power,mean=mu.v,variance=var.v,exact_mean=str(exact_mu),exact_variance=str(exact_var),variance_error=float(F(var.v)-exact_var),variance_relative_error=float((F(var.v)-exact_var)/exact_var),would_disable_moment=bool(var.v<=0 or mu.v<=0),named_counts=op.dict(),scope='Direct moment guard probe, not estimator-returned shortcut or FP certificate'))
 out.update(complete=True,cpu_including_imports=time.process_time()-START,cold_rules=rules.events,conditional_only=True)
 (P/'controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=True,cpu=out['cpu_including_imports'],methods=dict(Counter(r.get('method',r['mode']) for r in out['rows'])),probes=out['probes']),indent=2))
if __name__=='__main__':main()
