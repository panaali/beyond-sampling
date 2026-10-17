import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from cone_estimator import cone_estimate
from stable_volume import adaptive_mass,chebyshev_tail
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--geometry',required=True);a=p.parse_args()
start=time.process_time();data=dict(controls=[],models=[]);out=Path(__file__).with_name('adaptive_results.json')
for n in (2,4,8,16,20):
    actual=chebyshev_tail(.25,1.,n-2);known=(12*n+9)/3**(n-1)
    data['controls'].append(dict(name='uniform_interval_tail',order=n,actual=actual,known=known,difference=actual-known))
pv,b,meta=adaptive_mass(np.eye(4));data['controls'].append(dict(name='identity_orthant',estimate=pv,truth=1/16,bound=b,meta=meta))
r=np.eye(4);r[:,0]=(r[:,0]+r[:,1])/2;s=np.eye(4);s[:,1]=(s[:,0]+s[:,1])/2
pv,b,m=adaptive_mass(r);ps,bs,ms=adaptive_mass(s);data['controls'].append(dict(name='split_orthant',estimate=pv+ps,truth=1/16,bound=b+bs,meta=[m,ms]))
w=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
rng=np.random.default_rng(51001);models=[base]+[tuple(w+sig*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base) for sig in (.0025,.02)]
reference=json.loads(Path(__file__).with_name('cone_results.json').read_text())
for sig,model,ref in zip((0.,.0025,.02),models,reference['models']):
    value,stats=cone_estimate(model,geometry_path=a.geometry,max_cpu=110-(time.process_time()-start),integration_tolerance=1e-8)
    truth=ref['rules'][-1]['estimate']
    data['models'].append(dict(sigma=sig,estimate=value,reference_fixed_gl20=truth,difference=value-truth,stats=stats))
    data['total_cpu']=time.process_time()-start;out.write_text(json.dumps(data,indent=2)+'\n');print(sig,value,value-truth,stats['production_cpu'],stats['quadrature_nodes'],stats['integration_only_bound'],stats['order_counts'],flush=True)
