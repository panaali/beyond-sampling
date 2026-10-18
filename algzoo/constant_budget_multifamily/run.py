"""Frozen T23 cellwise reconstruction; no test-dependent optimization."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
HERE=Path(__file__).resolve().parent
WEIGHTS=Path('/workspace/work/t45_weights')
BUDGETS=(1,3,5,10,20)
FAMILIES=('fourier','position_polynomial','rank_values')

def features(x,family):
 p=x[:,:9].astype(np.float64);order=np.argsort(-p,axis=1,kind='stable');rank=np.take_along_axis(p,order,axis=1);m1,m2=rank[:,0],rank[:,1];p1,p2=order[:,0],order[:,1]
 cols=[m1,m2,x[:,9]];names=['m1','m2','x9']
 if family=='fourier':
  for prefix,value,pos in [('gap',m1-m2,p1),('m2',m2,p2)]:
   for f in range(1,5):
    for trig in(np.cos,np.sin):cols.append(value*trig(2*np.pi*f*pos/9));names.append(f'{prefix}_{trig.__name__}{f}')
 elif family=='position_polynomial':
  grid=np.linspace(-1,1,9)
  for prefix,value,pos in [('gap',m1-m2,p1),('m2',m2,p2)]:
   for degree in range(1,9):
    v=np.polynomial.legendre.legval(grid,[0]*degree+[1]);v-=v.mean();v/=np.sqrt(np.mean(v*v));cols.append(value*v[pos]);names.append(f'{prefix}_P{degree}')
 elif family=='rank_values':
  for r in range(2,9):cols.append(rank[:,r]);names.append(f'rank{r+1}')
  for r in range(9):cols.append(abs(rank[:,r]));names.append(f'absrank{r+1}')
  cols.append(abs(x[:,9]));names.append('absx9')
 else:raise ValueError(family)
 if family!='rank_values':cols.append(m2*np.cos(2*np.pi*(p1-p2)/9));names.append('m2_cos_position_difference')
 return np.column_stack(cols),names

def fit(phi,h):
 initial=np.linalg.lstsq(phi,h,rcond=None)[0];shape=initial.shape;calls=0
 def objective(flat):
  nonlocal calls
  calls+=1;pre=phi@flat.reshape(shape);error=np.maximum(pre,0)-h
  return float(np.mean(error*error)),(2*phi.T@(error*(pre>0))/error.size).ravel()
 initial_loss,initial_gradient=objective(initial.ravel());threshold=1e-7*max(1.,float(np.max(abs(initial_gradient))))
 beta=initial.ravel();segments=[]
 for segment in range(4):
  r=minimize(objective,beta,jac=True,method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-15,gtol=1e-8,maxls=50))
  beta=r.x;loss,gradient=objective(beta);gn=float(np.max(abs(gradient)))
  segments.append(dict(segment=segment,iterations=int(r.nit),objective_evals=int(r.nfev),success=bool(r.success),message=str(r.message),mse=loss,gradient_inf=gn))
  if gn<=threshold:break
 return beta.reshape(shape),dict(initial_mse=initial_loss,initial_gradient_inf=float(np.max(abs(initial_gradient))),threshold=threshold,segments=segments,converged=gn<=threshold,train_mse=loss,gradient_inf=gn,objective_evals=calls,worse_than_initial=loss>initial_loss+1e-12)

def wilson(success,n):
 p=success/n;z=1.959963984540054;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [float(c-r),float(c+r)]

def targets(rnn,x):
 chunks=[]
 with torch.inference_mode():
  for begin in range(0,len(x),8192):
   _,h=rnn(torch.from_numpy(x[begin:begin+8192]).unsqueeze(-1));chunks.append(h[0].numpy())
 return np.concatenate(chunks).astype(np.float64)

def main(H,seed):
 torch.set_num_threads(1);start=time.process_time();wall=time.perf_counter();path=WEIGHTS/f'2nd_argmax_{H}_10_1073741824_{seed}.pth';state=torch.load(path,map_location='cpu',weights_only=True)
 rnn=torch.nn.RNN(1,H,nonlinearity='relu',bias=False,batch_first=True);rnn.load_state_dict({k.removeprefix('rnn.'):v for k,v in state.items() if k.startswith('rnn.')});rnn.eval();output=state['linear.weight'].numpy().astype(np.float64)
 train=np.random.default_rng(72301).standard_normal((12000,10)).astype(np.float32);test=np.random.default_rng(72302).standard_normal((200000,10)).astype(np.float32);truth=np.argsort(test,axis=1,kind='stable')[:,-2]
 htrain=targets(rnn,train);htest=targets(rnn,test);correct=(htest@output.T).argmax(1)==truth;acc=float(correct.mean());variance=((htest-htest.mean(0))**2).sum(0)
 # Separate NumPy recurrence on the first1024 controls, with the same float32 weights.
 wi=state['rnn.weight_ih_l0'].numpy();wh=state['rnn.weight_hh_l0'].numpy();h=np.zeros((1024,H),dtype=np.float32)
 for t in range(10):h=np.maximum(h@wh.T+test[:1024,t,None]@wi.T,0)
 recurrence_error=float(np.max(abs(h-htest[:1024])));agreement=bool(np.array_equal((h@output.T).argmax(1),(htest[:1024]@output.T).argmax(1)))
 if not agreement:raise AssertionError('NumPy recurrence class disagreement')
 out=dict(H=H,T=10,seed=seed,training_sequences=2**30,weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),train_seed=72301,test_seed=72302,n_train=12000,n_test=200000,model_accuracy=acc,model_accuracy_ci=wilson(int(correct.sum()),len(test)),numpy_recurrence_max_error=recurrence_error,numpy_recurrence_class_agreement=agreement,rows=[],coefficients={},features={},compute=dict(model_hidden_forward_sequences=212000,independent_numpy_sequences=1024,threads=1))
 result_path=HERE/f'results_H{H}_seed{seed}.json';checkpoint_cpu=0.;previous_cpu=0.
 if result_path.exists():
  saved=json.loads(result_path.read_text());assert saved['H']==H and saved['seed']==seed and saved['weights_sha256']==out['weights_sha256']
  previous_cpu=saved['compute']['cpu_seconds'];out=saved
 out.setdefault('attempts',[]).append(dict(starting_completed_cells=len(out['rows']),prior_recorded_cpu=previous_cpu))
 out['status']='partial'
 def save():
  nonlocal checkpoint_cpu
  t=time.process_time();out['compute'].update(cpu_seconds=previous_cpu+time.process_time()-start,current_attempt_cpu=time.process_time()-start,wall_seconds=time.perf_counter()-wall,checkpoint_cpu=checkpoint_cpu,objective_calls=sum(r['solver']['objective_evals'] for r in out['rows']))
  result_path.write_text(json.dumps(out,indent=2)+'\n');checkpoint_cpu+=time.process_time()-t
 save()
 for family in FAMILIES:
  phi,names=features(train,family);phi_test,_=features(test,family);out['features'][family]=names;last=None
  for K in BUDGETS:
   key=f'{family}_{K}'
   if key in out['coefficients']:
    old_beta=np.asarray(out['coefficients'][key]);last=(np.maximum(phi_test[:,:K]@old_beta,0)@output.T).argmax(1)==truth
    continue
   if time.process_time()-start>45:
    out['attempts'][-1]['bounded_return']=True;save();print('BOUNDED partial',len(out['rows']),flush=True);return
   t=time.process_time();beta,solver=fit(phi[:,:K],htrain);prediction=np.maximum(phi_test[:,:K]@beta,0);error=((prediction-htest)**2).sum(0);r2=np.where(variance>1e-9,1-error/np.maximum(variance,1e-9),(error<1e-6).astype(float));good=(prediction@output.T).argmax(1)==truth;rate=float(good.mean())
   delta=None
   if last is not None:
    d=good.astype(float)-last.astype(float);se=d.std(ddof=1)/np.sqrt(len(d));delta=dict(mean=float(d.mean()),ci=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)])
   row=dict(family=family,budget=K,fitted_scalars=int(beta.size),replacement_accuracy=rate,accuracy_ci=wilson(int(good.sum()),len(test)),gap_recovered=(rate-.1)/(acc-.1),passes_reconstruction_threshold=rate>=.1+.9*(acc-.1),r2=r2.tolist(),n_r2_at_least_0_8=int((r2>=.8).sum()),mean_r2=float(r2.mean()),change_from_previous_budget=delta,solver=solver,fit_score_cpu=time.process_time()-t)
   out['rows'].append(row);out['coefficients'][f'{family}_{K}']=beta.tolist();last=good;save();print(H,seed,family,K,rate,solver['converged'],solver['gradient_inf'],flush=True)
 out['status']='complete';save();print('CPU',out['compute']['cpu_seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--H',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.H,a.seed)
