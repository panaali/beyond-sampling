import json,time
from pathlib import Path
import numpy as np
from ..accounting import Audit
from .geometry import determinant,GATE_IDS,OUTPUT_IDS,RANK_IDS,ORTHANT_IDS
from .bounded_rank import bounded_rank,AmbiguousRank
HERE=Path(__file__).resolve().parent;start=time.process_time();rng=np.random.default_rng(52061);errors=[]
for i in range(20):
 a=rng.normal(size=(6,6))
 if i%2:a[:,5]=a[:,0]+1e-8*a[:,5]
 error=abs(determinant(a,Audit())-np.linalg.det(a));assert error<1e-10;errors.append(error)
for value,expected in ((0.,5),(1e-6,6),(1e-9,None)):
 try:r=bounded_rank(np.diag([1.,1.,1.,1.,1.,value]),Audit());assert expected is not None and r==expected
 except AmbiguousRank:assert expected is None
out=dict(seed=52061,determinant_cases=20,max_det_error=max(errors),rank_controls='rank5,rank6,threshold ambiguity passed',constraint_ids={k:sorted(v) for k,v in dict(gate=GATE_IDS,output=OUTPUT_IDS,rank=RANK_IDS,orthant=ORTHANT_IDS).items()},cpu_seconds=time.process_time()-start)
(HERE/'primitive_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
