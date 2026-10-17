"""Bounded additional T4 levels, no repeated MC or trained weights download."""
import argparse,importlib.util,json,time
from pathlib import Path
from run_t3 import load
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--engine',default=str(HERE.parent/'small_h4_4'/'estimator.py'));ap.add_argument('--orders',type=int,nargs='+',default=[64]);ap.add_argument('--offset',type=float,default=0);ap.add_argument('--mode',default='octants');ap.add_argument('--subdivision',type=int,default=1);ap.add_argument('--output',default='refinement_t4.json');a=ap.parse_args()
spec=importlib.util.spec_from_file_location('t51_circle_refine',a.engine);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
model=load(4);result={'rows':[],'engine':a.engine,'offset':a.offset,'mode':a.mode,'subdivision':a.subdivision};start=time.process_time()
for n in a.orders:
 try:
  p,s=engine.angular_estimate(model,order=n,offset=a.offset,boundary_mode=a.mode,subdivision=a.subdivision,max_cpu=110-(time.process_time()-start))
  result['rows'].append(dict(accuracy=p,**s));print(n,p,s['cpu_seconds'],flush=True)
 except (ArithmeticError,ValueError,TimeoutError) as error:
  result['failure']=dict(order=n,error=str(error),kind=type(error).__name__);break
 result['total_cpu']=time.process_time()-start
 (HERE/a.output).write_text(json.dumps(result,indent=2)+'\n')
result['total_cpu']=time.process_time()-start
(HERE/a.output).write_text(json.dumps(result,indent=2)+'\n')
