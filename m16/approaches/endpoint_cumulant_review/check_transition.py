"""Fixed prior Gaussian laws and thin-wedge sensitivity; no new models."""
import time
START=time.process_time()
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import full_covariance_bounded as B
import transition_orthant as N
HERE=Path(__file__).resolve().parent

def st():return dict(quad_calls=0,quad_evaluations=0,quad_error_estimate_sum=0.,singular_limits=0,outside_unit_corrections=[],singular_orthant_error_bounds=[],analytic_tail_bound_sum=0.)
def main():
    laws=[];c=json.loads((HERE/'controls.json').read_text())
    for r in c['cases']:laws.append((r['name'],r['cumulants']))
    for name,mu,cov in [('zero',[0,0,0],np.zeros((3,3))),('duplicate',[1,1,0],np.zeros((3,3))),('opposite',[.5,0,-.5],np.outer([1,0,-1],[1,0,-1])),('same_slope',[0,1,2],np.ones((3,3)))]:laws.append((name,{'1':mu,'2':cov}))
    for g in json.loads((HERE/'trained_t3.json').read_text())['endpoint_cases']:
        for r in g['conditional']:laws.append((str(g['atom_separated'])+str(r['target']),r['cumulants']))
    for r in json.loads((HERE/'gate_mixture.json').read_text())['components']:
        if r['fixed_winner'] is None:laws.append((str(r['gate_cell'])+str(r['order']),r['cumulants']))
    old=B.orthant;rows=[]
    for name,K in laws:
        B.orthant=old;a=B.winners(K['1'],K['2'])
        B.orthant=N.orthant;b=B.winners(K['1'],K['2'])
        diff=float(np.max(abs(np.array(a['q'])-b['q'])));rows.append(dict(name=name,old=a,new=b,difference=diff))
    B.orthant=old
    extreme=[]
    for a in [-100.,0.,100.]:
        for b in [-100.,0.,100.]:
            s=st();p=N.orthant([a,b],np.eye(2),[False,False],1e-12,s);truth=float(ndtr(a)*ndtr(b));assert abs(p-truth)<1e-12
            extreme.append(dict(a=a,b=b,p=p,truth=truth,stats=s))
    assert len(rows)==27
    result=dict(complete=True,rows=rows,extreme=extreme,max_prior_difference=max(r['difference'] for r in rows),cpu_seconds=time.process_time()-START,
        pinned_orthant_hash=N.source_hash,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'check_transition.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['max_prior_difference'],result['cpu_seconds'])
if __name__=='__main__':main()
