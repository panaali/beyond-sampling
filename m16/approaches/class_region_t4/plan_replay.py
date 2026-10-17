"""Exact count under the inherited named-operation convention, not full cost."""
import argparse,json,time
import numpy as np
from . import runner as R
p=argparse.ArgumentParser();p.add_argument('kind',choices=['quadratic','nearest']);a=p.parse_args();start=time.process_time();data=np.load(R.HERE/(a.kind+'_exact_mass_calls.npz'));rays=data['rays'];thin=wide=skip=0
for ray in rays:
    B=np.linalg.inv(ray.T);S=B@B.T;diag=np.sqrt(np.diag(S));C=S/np.outer(diag,diag)
    if abs(np.linalg.det(ray))<1e-12:skip+=1;continue
    maximum=max(abs(C[i,j]) for i in range(4) for j in range(i+1,4))
    if maximum>.95:thin+=1
    else:wide+=1
plans={str(m):len(rays)*156+152*(thin*48+wide*12)*m for m in (1,2,4)}
prior=R.HERE/(a.kind+'_mass_x2.json')
if prior.exists():
    x=json.loads(prior.read_text())
    if x['complete']:assert x['partial_named_ops']==plans['2']
out=dict(kind=a.kind,thin=thin,wide=wide,determinant_cutoff=skip,plans=plans,cpu=time.process_time()-start,named_op_cap=200000000,quadrupled_exceeds_cap=plans['4']>200000000,exact_under_partial_named_convention=True,portable_cost_certified=False)
(R.HERE/(a.kind+'_replay_plan.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
