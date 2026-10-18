"""Freeze the actual unchanged source branch before MP attribution."""
import json,time
import numpy as np
from review import HERE,SOURCE,HASHES,load_source,Ops
start=time.process_time()
source=load_source();saved=json.loads((SOURCE/'negative_mass_fixture.json').read_text())
rays=np.asarray(saved['rays']);B=np.linalg.inv(rays.T);S=B@B.T
d=np.sqrt(np.diag(S));C=S/np.outer(d,d)
rmax=float(max(abs(C[i,j]) for i in range(4) for j in range(i+1,4)))
determinant=float(np.linalg.det(rays));name='thin' if rmax>.95 else 'wide'
assert name=='wide' and abs(determinant)>=1e-12
ops=Ops();value=source.orthant(rays,ops)
assert abs(value-saved['source_value'])<1e-16
result=dict(index=saved['index'],rmax=rmax,determinant=determinant,determinant_guard=1e-12,
 rule_threshold=.95,rule_name=name,order=12,power=2,rounded_rule=[[float(x) for x in row] for row in source._RULES[name]],
 source_value=value,source_named_ops=ops.n,source_hashes=HASHES,cpu_seconds=time.process_time()-start)
(HERE/'rule_metadata.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rounded_rule'}))
