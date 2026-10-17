"""H2T4 benchmark using T51-owned circle driver; no duplicated geometry."""
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
from run_t3 import load
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--engine',default=str(HERE.parent/'small_h4_4'/'estimator.py'));args=parser.parse_args()
spec=importlib.util.spec_from_file_location('t51_circle',args.engine);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
start=time.process_time();model=load(4);report=dict(rows=[],engine=args.engine,checkpoint='verified H2T4 seed0',failures=[])
# Coarse validation and inclusive sampling CPU baseline; outcomes are not used to select nodes.
begin=time.process_time();x=np.random.default_rng(47004).normal(size=(65536,4));hits=engine.forward(model,x).argmax(1)==np.argsort(x,axis=1)[:,-2];baseline=time.process_time()-begin
report['mc']=dict(n=len(x),seed=47004,accuracy=float(hits.mean()),se=float(np.sqrt(hits.mean()*(1-hits.mean())/len(x))),cpu_seconds=baseline)
for order in (2,4,8,16,32):
 try:
  p,s=engine.angular_estimate(model,order=order,max_cpu=100-(time.process_time()-start))
 except (ArithmeticError,ValueError,TimeoutError) as error:
  report['failures'].append(dict(order=order,type=type(error).__name__,message=str(error)));break
 equivalents=s['cpu_seconds']/baseline*len(x)
 report['rows'].append(dict(accuracy=p,cpu_equivalent_samples=equivalents,sampling_se_at_equal_cpu=float(np.sqrt(p*(1-p)/equivalents)),**s))
 report['total_cpu']=time.process_time()-start
 (HERE/'results_t4.json').write_text(json.dumps(report,indent=2)+'\n')
 print(order,p,s['cpu_seconds'],flush=True)
 if time.process_time()-start>105:break
report['total_cpu']=time.process_time()-start
(HERE/'results_t4.json').write_text(json.dumps(report,indent=2)+'\n')
print('failures',report['failures'],flush=True)
