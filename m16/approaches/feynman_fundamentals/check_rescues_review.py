import sys,time,json,hashlib
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).parent;start=time.process_time()
T37=Path('/workspace/work/recover-small_h2_56/m16/approaches/hidden_reference_review')
sys.path.insert(0,str(T37))
from probe import reference_model,recurrence
from support_rescue import clipped
saved=json.loads((T37/'results.json').read_text())
state=torch.load(T37.parent/'face_surplus_review/data/H16_T10_original.pth',map_location='cpu',weights_only=True)
base=[state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')];base[0]=base[0].ravel()
ref=reference_model();rows=[]
for seed in (370202,370203):
 x=np.random.default_rng(seed).standard_normal((512,10));r=np.zeros((512,11,22))
 for t in range(10):r[:,t+1]=np.maximum(r[:,t]@ref[1].T+x[:,t,None]*ref[0],0)
 for old in saved['models']:
  sigma=old['sigma'];b=np.array(old['alignment']);rng=np.random.default_rng(1)
  wi,w,wo=tuple(v+sigma*np.sqrt(np.mean(v*v))*rng.standard_normal(v.shape) for v in base)
  hbar=r@b.T;e=np.zeros((512,16));hc=e.copy();ha=e.copy()
  for t in range(10):
   z=hbar[:,t]@w.T+x[:,t,None]*wi;gate=z>0
   if t==9:rescue=np.maximum((hbar[:,t]+e)@w.T+x[:,t,None]*wi,0)
   delta=np.maximum(z,0)-hbar[:,t+1]
   e=delta+gate*(e@w.T)
   hc=gate*np.maximum(hc@w.T+x[:,t,None]*wi,0)
   ha=np.maximum(ha@w.T+x[:,t,None]*wi,0)
  own={'linear':(hbar[:,-1]+e)@wo.T,'last_exact':rescue@wo.T,'actual':ha@wo.T}
  other,_=recurrence((wi,w,wo),b,x,r);oc,_,_=clipped((wi,w,wo),b,x,r)
  differences={k:float(np.max(abs(v-other[k]))) for k,v in own.items()};differences['clipped']=float(np.max(abs(hc@wo.T-oc)))
  labels={k:int(np.sum(v.argmax(1)!=other[k].argmax(1))) for k,v in own.items()};labels['clipped']=int(np.sum((hc@wo.T).argmax(1)!=oc.argmax(1)))
  assert max(differences.values())<1e-9
  rows.append(dict(seed=seed,sigma=sigma,max_logit_differences=differences,label_disagreements=labels))
result={'T37':rows,'T37_cpu_seconds':time.process_time()-start,'calibration_sha256':hashlib.sha256((T37/'results.json').read_bytes()).hexdigest()}
T91=Path('/workspace/work/recover-small_h2_78/m16/approaches/stratum_transport');sys.path.insert(0,str(T91))
from event_volume import EventInterfaceOperator
state=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth',map_location='cpu',weights_only=True)
model=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=(model[0].ravel(),*model[1:])
author=json.loads((T91/'split_loglinear_results.json').read_text());refs=json.loads((T91/'fourth_step_results.json').read_text())['rows'];rows=[]
for n,event,swap in ((16,False,False),(32,False,False),(16,True,True)):
 tick=time.process_time();P=np.array([[0.,1.],[1.,0.]])
 m=(P@model[0],P@model[1]@P,model[2]@P) if swap else model
 op=EventInterfaceOperator(m,n,'loglinear',event);s=op.initial();steps=[]
 for t in range(4):
  s,output=op.step(s);steps.append(dict(masses=s.masses(),output=output.tolist(),sides=[float(v.area.sum()) for v in s.side_states]))
 reference=next(v['reference'] for v in refs if v['model']=='trained_H2T8' and v['grid']==n)
 prior=next(v for v in author['rows'] if v['model']=='trained_H2T8' and v['grid']==n)
 error=float(np.max(abs(output-reference['output_masses'])))
 difference=float(np.max(abs(output-prior['steps'][-1]['output_masses'])))
 rows.append(dict(grid=n,event_cut=event,coordinate_swap=swap,endpoint_error=error,difference_from_event_on=difference,steps=steps,cpu_seconds=time.process_time()-tick,work=op.report()))
 print('T91',n,event,swap,error,difference,flush=True)
 assert time.process_time()-start<19
result.update(T91=rows,total_cpu_seconds=time.process_time()-start,new_random_input_streams=0,T37_replayed_sequences=2048,T91_model_input_sampling=0)
(OUT/'rescues_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
print('T37',result['T37'],'CPU',result['total_cpu_seconds'],flush=True)
