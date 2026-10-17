"""Frozen two-model wrapper diagnosis and four analytic tie controls."""
import time
START=time.process_time()
import hashlib,json,sys,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'snapshot'),str(HERE/'dependencies'/'understand_m43'),str(HERE/'dependencies'/'arc_m43_repro')]
import rec112 as R
import mine as M
import est75 as E
import exact as EX
from m43 import Ops

def equality_install():
 def eq(self,other):R.C['n']+=1;return self.x==R.val(other)
 def ne(self,other):R.C['n']+=1;return self.x!=R.val(other)
 R.Num.__eq__=eq;R.Num.__ne__=ne

def equality_remove():
 for name in ('__eq__','__ne__'):
  if name in R.Num.__dict__:delattr(R.Num,name)

def main():
 out=dict(primitives={},models=[],analytic=[],scope='Two diagnostic models and exact constant-label fixtures only; no full counter repair or trained-truth rejection.')
 for patched in (False,True):
  if patched:equality_install()
  out['primitives'][str(patched)]=[dict(x=x,eq_zero=R.Num(x)==0,ne_zero=R.Num(x)!=0) for x in (0.,1.)]
 equality_remove()
 for name,wi,wo in [('alloff',0.,[0.,0.,0.]),('zero-readout',1.,[0.,0.,0.]),('identical-readout',1.,[1.,1.,1.]),('first-dominant',1.,[1.,0.,-1.])]:
  model=(np.array([wi]),np.zeros((1,1)),np.array(wo).reshape(3,1));row=dict(name=name,analytic_truth=1/3)
  for label,fn in [('reference_ordering',lambda:EX.exact_acc(model)),('reference_octants',lambda:M.exact_mine(model)),('recipe',lambda:E.est75(model,Ops(),**E.RECIPE112)),('reimplementation',lambda:R.rec112(R.as_lists(model)))]:
   try:row[label]=float(fn())
   except Exception as exc:row[label+'_failure']=repr(exc)
  out['analytic'].append(row)
 import torch
 for H,zs,copy,path,index in [(8,2,3,'/workspace/work/t45_weights/2nd_argmax_8_3_1073741824_2.pth',14),(4,1,2,'/workspace/work/t114_checkpoints/2nd_argmax_4_3_1073741824_1.pth',8)]:
  sd=torch.load(path,map_location='cpu',weights_only=True);assert set(sd)=={'rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'}
  base=(sd['rnn.weight_ih_l0'].numpy().astype(float)[:,0],sd['rnn.weight_hh_l0'].numpy().astype(float),sd['linear.weight'].numpy().astype(float));model=M.noisy_abs(base,4,.05,89)[copy];row=dict(H=H,zoo=zs,noise_seed=89,copy=copy,checkpoint=path,checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
  row['recipe']=float(E.est75(model,Ops(),**E.RECIPE112));row['float_reimplementation']=R.rec112(R.as_lists(model));equality_remove();row['original_wrapped'],row['original_count']=R.counted(model);equality_install();row['equality_repaired'],row['equality_repaired_count']=R.counted(model);equality_remove();row['saved']=json.loads((HERE/'snapshot'/'out'/'count.json').read_text())[str(H)]['rows'][index];out['models'].append(row)
 out['cpu']=time.process_time()-START;(HERE/'semantics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
