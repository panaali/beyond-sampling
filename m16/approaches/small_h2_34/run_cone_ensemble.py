import argparse,json,time
from pathlib import Path
import numpy as np
from run_t3 import load
from cone_estimator import calculate,forward
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--geometry');parser.add_argument('--mass');a=parser.parse_args()
base=load(4);rng=np.random.default_rng(47001);models=[base]+[tuple(w+.05*rng.normal(size=w.shape) for w in base) for _ in range(8)]
start=time.process_time();rows=[];output=dict(noise_seed=47001,noise_sigma_absolute=.05,rows=rows)
for i,model in enumerate(models):
    try:
        p,s,(g,cells,rays,vol)=calculate(model,a.geometry,a.mass,retain=True)
        begin=time.process_time();refs=np.array([vol.mass(r,20) for r in rays]);ref=float(refs[:,0].sum());refbound=float(refs[:,1].sum());refcpu=time.process_time()-begin
        gate_rays=g.simplices([v for v,B in cells]);gate_det=float(np.abs(np.linalg.det(gate_rays)).sum())
        begin=time.process_time();x=np.random.default_rng(47100+i).normal(size=(65536,4));hit=forward(model,x).argmax(1)==np.argsort(x,axis=1)[:,-2];baseline=time.process_time()-begin
        equivalents=s['cpu_seconds']/baseline*len(x);mc=float(hit.mean());se=float(np.sqrt(p*(1-p)/len(x)))
        rows.append(dict(copy=i,accuracy=p,reference=ref,difference=p-ref,reference_integration_only_bound=refbound,
            reference_cpu=refcpu,stats=s,gate_determinant_sum=gate_det,mc=mc,mc_se=se,mc_z=(mc-p)/se,
            baseline_cpu=baseline,cpu_equivalent_samples=equivalents,sampling_mse_at_equal_cpu=p*(1-p)/equivalents,
            numerical_mse_ratio=(p-ref)**2/(p*(1-p)/equivalents)))
        print(i,p,s['cpu_seconds'],refbound,flush=True)
    except (ArithmeticError,ValueError,TimeoutError) as error:
        rows.append(dict(copy=i,error=str(error),error_type=type(error).__name__))
    output['total_cpu']=time.process_time()-start
    (HERE/'cone_ensemble_t4.json').write_text(json.dumps(output,indent=2)+'\n')
    if time.process_time()-start>110:break
