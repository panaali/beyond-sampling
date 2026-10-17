import json,time
from pathlib import Path
import numpy as np
from families import run
from m43 import Wi0,Wh0,Wo0,noisy_models
from tie_repair import activate
start=time.process_time();activate();rows=[]
for name,wo in [('zero',np.zeros_like(Wo0)),('identical',np.tile(Wo0[0],(3,1))),('duplicate_top',np.array([[1.]*4,[1.]*4,[0.]*4])),('duplicate_lower',np.array([[0.]*4,[1.]*4,[1.]*4]))]:
 for kind in('old','quadratic','nearest'):
  r=run((Wi0,Wh0,wo),kind,'exact');r.update(name=name,kind=kind)
  # duplicate_top always predicts0: hiddenzero also0. duplicate_lower predicts1 excepthiddenzero.
  if name!='duplicate_lower':r.update(reference=1/3,error=r['value']-1/3)
  rows.append(r)
original=json.loads(Path(__file__).with_name('results.json').read_text());trained=[]
for i,m in enumerate([(Wi0,Wh0,Wo0)]+noisy_models(k=4,sigma=.05,seed=80)):
 for mode in('exact','recipe'):
  for kind in('old','quadratic','nearest'):
   r=run(m,kind,mode);old=next(z for z in original['trained'][i]['rows'] if z['kind']==kind and z['mode']==mode)
   r.update(model=i,kind=kind,mode=mode,change=r['value']-old['value']);trained.append(r)
# Cell-level independent exact orthant witnesses: ray-matrix=identity, allhiddenpositive, knownwinner.
from families import engine,Ops
c=engine.Cone(np.eye(3),np.eye(3));A=np.eye(3);g=np.ones(3);controls=[]
for wo in [np.zeros((3,3)),np.ones((3,3)),np.array([[1,1,1],[1,1,1],[0,0,0]]),np.array([[0,0,0],[1,1,1],[1,1,1]])]:
 winner=int(np.argmax(wo@np.ones(3)))
 for truth in range(3):
  v=engine._cell_error(c,g,A,wo,truth,Ops());ref=0 if truth==winner else.125
  controls.append(dict(winner=winner,truth=truth,value=v,reference=ref));assert abs(v-ref)<1e-15
out=dict(ties=rows,trained=trained,cell_controls=controls,cpu=time.process_time()-start)
Path(__file__).with_name('repair_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu=out['cpu'],tie_errors=[z.get('error') for z in rows],max_trained_change=max(abs(z['change']) for z in trained),cell_controls=len(controls))))
