"""One frozen reversible training-coordinate rescue; original files immutable."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from run import HERE,WEIGHTS,features,targets,wilson

def objective(phi,h,beta):
 pre=phi@beta;error=np.maximum(pre,0)-h
 return float(np.mean(error*error)),2*phi.T@(error*(pre>0))/error.size

def rescue(phi,h,beta,threshold):
 initial=beta.copy();q,r=np.linalg.qr(phi,mode='reduced');rank=int(np.linalg.matrix_rank(r));K=phi.shape[1]
 if rank!=K:raise ArithmeticError('full-rank QR required; no truncation allowed')
 root=np.sqrt(len(phi));s=q*root;gamma=r@beta/root;shape=gamma.shape;calls=0
 factor_error=float(np.max(abs(q@r-phi)));transform_error=float(np.max(abs(s@gamma-phi@beta)));initial_loss,_=objective(phi,h,beta)
 if transform_error>1e-9*max(1.,float(np.max(abs(phi@beta)))):raise ArithmeticError('coordinate transform failed floating control')
 def fn(flat):
  nonlocal calls
  calls+=1;loss,grad=objective(s,h,flat.reshape(shape));return loss,grad.ravel()
 segments=[]
 for i in range(4):
  result=minimize(fn,gamma.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-15,gtol=1e-12,maxls=50));gamma=result.x.reshape(shape);beta=np.linalg.solve(r,gamma*root);loss,gradient=objective(phi,h,beta);gn=float(np.max(abs(gradient)))
  segments.append(dict(segment=i,iterations=int(result.nit),objective_evals=int(result.nfev),success=bool(result.success),message=str(result.message),train_mse=loss,original_gradient_inf=gn,transformed_gradient_inf=float(np.max(abs(result.jac)))))
  if gn<=threshold:break
 return beta,dict(initial_mse=initial_loss,train_mse=loss,original_gradient_inf=gn,threshold=threshold,converged=gn<=threshold,worse_than_initial=loss>initial_loss+1e-12,condition_R=float(np.linalg.cond(r)),rank=rank,factorization_error=factor_error,initial_transform_error=transform_error,final_transform_error=float(np.max(abs(s@gamma-phi@beta))),objective_evals=calls,original_gradient_evaluations=1+len(segments),segments=segments)

def main(H,seed):
 files=sorted(HERE.glob('results_H*_seed*.json'));assert len(files)==6 and all(len(json.loads(p.read_text())['rows'])==15 for p in files),'finish all original90 fits first'
 torch.set_num_threads(1);start=time.process_time();original_path=HERE/f'results_H{H}_seed{seed}.json';old=json.loads(original_path.read_text());sha=hashlib.sha256(original_path.read_bytes()).hexdigest();dest=HERE/f'qr_H{H}_seed{seed}.json'
 state=torch.load(WEIGHTS/f'2nd_argmax_{H}_10_1073741824_{seed}.pth',map_location='cpu',weights_only=True);rnn=torch.nn.RNN(1,H,nonlinearity='relu',bias=False,batch_first=True);rnn.load_state_dict({k.removeprefix('rnn.'):v for k,v in state.items() if k.startswith('rnn.')});rnn.eval();output=state['linear.weight'].numpy().astype(np.float64)
 train=np.random.default_rng(72301).standard_normal((12000,10)).astype(np.float32);test=np.random.default_rng(72302).standard_normal((200000,10)).astype(np.float32);truth=np.argsort(test,axis=1,kind='stable')[:,-2];htrain=targets(rnn,train);htest=targets(rnn,test);variance=((htest-htest.mean(0))**2).sum(0)
 out=json.loads(dest.read_text()) if dest.exists() else dict(H=H,seed=seed,original_sha256=sha,rows=[],coefficients={},attempts=0,cpu_seconds=0.)
 assert out['original_sha256']==sha;prior=out['cpu_seconds'];out['attempts']+=1;out['status']='partial'
 def save():
  out['cpu_seconds']=prior+time.process_time()-start;out['hidden_forward_sequences']=212000*out['attempts'];dest.write_text(json.dumps(out,indent=2)+'\n')
 for row in old['rows']:
  family,K=row['family'],row['budget'];key=f'{family}_{K}'
  if row['solver']['converged'] or key in out['coefficients']:continue
  if time.process_time()-start>45:save();print('BOUNDED',H,seed,len(out['rows']),flush=True);return
  phi,_=features(train,family);phi=phi[:,:K];beta=np.asarray(old['coefficients'][key]);t=time.process_time();fixed,solver=rescue(phi,htrain,beta,row['solver']['threshold']);phi_test,_=features(test,family);prediction=np.maximum(phi_test[:,:K]@fixed,0);error=((prediction-htest)**2).sum(0);r2=np.where(variance>1e-9,1-error/np.maximum(variance,1e-9),(error<1e-6).astype(float));good=(prediction@output.T).argmax(1)==truth;rate=float(good.mean())
  oldgood=(np.maximum(phi_test[:,:K]@beta,0)@output.T).argmax(1)==truth;delta=good.astype(float)-oldgood.astype(float);se=delta.std(ddof=1)/np.sqrt(len(delta));out['rows'].append(dict(family=family,budget=K,original_accuracy=row['replacement_accuracy'],replacement_accuracy=rate,accuracy_ci=wilson(int(good.sum()),len(test)),gap_recovered=(rate-.1)/(old['model_accuracy']-.1),r2=r2.tolist(),n_r2_at_least_0_8=int((r2>=.8).sum()),accuracy_change=float(delta.mean()),paired_change_ci=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],solver=solver,fit_score_cpu=time.process_time()-t));out['coefficients'][key]=fixed.tolist();save();print(H,seed,family,K,rate,solver['converged'],solver['original_gradient_inf'],flush=True)
 out['status']='complete';save();assert hashlib.sha256(original_path.read_bytes()).hexdigest()==sha
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--H',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.H,a.seed)
