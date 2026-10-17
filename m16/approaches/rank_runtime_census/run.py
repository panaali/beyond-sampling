"""Frozen geometry/native census, validating all ordered rays without mass work."""
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from observer import HERE,reset,report
PRIOR=HERE.parent/'plackett_cost_review';AUTHOR=Path('/workspace/work/recover-small_h2_910/m16/approaches')
PINS={PRIOR/'inventory.py':'25fbf143de8d93e6a9b42e0d38754b0ea8369b07af14cb65771e6586ceb800c7',
 AUTHOR/'leaf_mass_review/error_captured.npz':'15f9e5e483c69325c10fcbe4eaaa80f91503b177070ccec1ff2569a11c9436b6',
 AUTHOR/'recipe_mass_separation/error_captured.npz':'1a4ee131e9ab252a46828a1d1d2985b9d397ec1b3a626a71ddb3c00535682c44',
 Path('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_0.pth'):'b771f40fcae826e010b2e89fb698e2ec35026119bc9f66d3ff046dba537b07da'}
for p,sha in PINS.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,str(p)
sys.path.insert(0,str(PRIOR));from inventory import load,Ops

def execute(model,mode,first,last,expected=None):
 reset();E,_,_=load(False);halves=E.halves(4);cold=report();reset()
 E._HALVES[4]=halves[first:last];E.STATS.update({k:0 for k in E.STATS});count=0;rays=[]
 def record(R,o):
  nonlocal count
  if expected is not None:assert np.array_equal(R,expected[count]),(mode,first,last,count)
  else:rays.append(np.array(R,copy=True))
  count+=1;E.STATS['simplices']+=1;return 0.
 E.orthant=record;ops=Ops();value=E.estT(model,ops,mode=mode)
 assert value==1.
 if expected is not None:assert count==len(expected)
 observed=report();assert all(x['failures']==0 for x in observed['kernels'])
 assert observed['gesdd'][9]==observed['gesdd'][10]==observed['nested'][3]==observed['nested'][7]==observed['nested'][11]==0
 assert sum(x['calls'] for x in observed['kernels'] if x['kernel'] in ('dlasq5','dlasq6'))==observed['nested'][8]
 return dict(mode=mode,first=first,last=last,legacy_geometry=ops.n,stats=dict(E.STATS),mass_callbacks=count,native=observed,cold=cold),rays

def controls(observed):
 start=time.process_time();models=[('zero',(np.zeros(2),np.zeros((2,2)),np.zeros((4,2)))),('rational',(np.array([1.,-1.]),np.array([[0.,1.],[1.,0.]]),np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])))];rows=[]
 baseline=json.loads((HERE/'geometry_baseline.json').read_text())['rows'] if observed else None
 for mode in ('exact','recipe108'):
  for name,model in models:
   old=next(x for x in baseline if x['name']==name and x['result']['mode']==mode) if observed else None
   result,rays=execute(model,mode,0,96,np.asarray(old['rays']) if observed else None)
   if observed:
    for key in ('legacy_geometry','stats','mass_callbacks'):assert result[key]==old['result'][key],(name,mode,key)
    assert name=='zero' or result['native']['gesdd'][0]>0
   else:assert result['native']['gesdd'][0]==0
   rows.append(dict(name=name,result=result,rays=old['rays'] if observed else np.asarray(rays).tolist()))
 out=dict(rows=rows,cpu_seconds=time.process_time()-start)
 (HERE/('geometry_observed.json' if observed else 'geometry_baseline.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(observed=observed,cases=len(rows),cpu=out['cpu_seconds'])))

def batch(mode,index):
 start=time.process_time();assert mode in ('exact','recipe108') and 0<=index<8
 import torch
 torch.set_num_threads(1);s=torch.load('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_0.pth',map_location='cpu',weights_only=True)
 model=(s['rnn.weight_ih_l0'].numpy().astype(float)[:,0],s['rnn.weight_hh_l0'].numpy().astype(float),s['linear.weight'].numpy().astype(float))
 folder='leaf_mass_review' if mode=='exact' else 'recipe_mass_separation';archive=np.load(AUTHOR/folder/'error_captured.npz');parents=archive['parent_ids'];selected=(parents>=12*index)&(parents<12*(index+1))
 result,_=execute(model,mode,12*index,12*(index+1),archive['rays'][selected])
 prior=HERE.parent/('plackett_cost_review' if mode=='exact' else 'recipe_cost_review')/f'geometry_{index}.json';old=json.loads(prior.read_text())
 for key in ('legacy_geometry','stats','mass_callbacks'):assert result[key]==old[key],key
 # Existing source ledger independently identifies one query plus execution per rank.
 old_rank=sum(x['count'] for x in old['native']['opaque_calls'] if x['key'][1]=='np.linalg.matrix_rank')
 assert result['native']['gesdd'][2]==result['native']['gesdd'][3]==old_rank
 result.update(batch=index,cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pins={str(k):v for k,v in PINS.items()},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (HERE/f'{mode}_{index}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('mode','batch','cpu_seconds','mass_callbacks','stats','legacy_geometry')}))

if __name__=='__main__':
 try:
  phase=sys.argv[1]
  if phase in ('baseline','observed'):controls(phase=='observed')
  else:batch(phase,int(sys.argv[2]))
 except Exception as exc:
  (HERE/('failure_'+'_'.join(sys.argv[1:])+'.json')).write_text(json.dumps(dict(failure=repr(exc)),indent=2)+'\n');raise
