"""Frozen14-cell separable rescue; original and QR artifacts stay immutable."""
import argparse,hashlib,json,time
import numpy as np
import torch
from run import HERE,WEIGHTS,features,targets,wilson
from correction import fit

def main(H,seed):
 start=time.process_time();torch.set_num_threads(1);op=HERE/f'results_H{H}_seed{seed}.json';qp=HERE/f'active_H{H}_seed{seed}.json';old=json.loads(op.read_text());q=json.loads(qp.read_text());hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (op,qp)};assert q['source_hashes'][op.name]==hashes[op.name]
 for name,sha in q['source_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
 assert (H,seed)==(32,2) and {(r['family'],r['budget']) for r in q['rows'] if not r['solver']['converged']}=={('fourier',10),('fourier',20),('position_polynomial',10)}
 wp=WEIGHTS/f'2nd_argmax_{H}_10_1073741824_{seed}.pth';assert hashlib.sha256(wp.read_bytes()).hexdigest()==old['weights_sha256'];state=torch.load(wp,map_location='cpu',weights_only=True);rnn=torch.nn.RNN(1,H,nonlinearity='relu',bias=False,batch_first=True);rnn.load_state_dict({k.removeprefix('rnn.'):v for k,v in state.items() if k.startswith('rnn.')});rnn.eval();output=state['linear.weight'].numpy().astype(np.float64)
 train=np.random.default_rng(72301).standard_normal((12000,10)).astype(np.float32);test=np.random.default_rng(72302).standard_normal((200000,10)).astype(np.float32);truth=np.argsort(test,axis=1,kind='stable')[:,-2];htrain=targets(rnn,train);htest=targets(rnn,test);variance=((htest-htest.mean(0))**2).sum(0)
 dest=HERE/f'correction_H{H}_seed{seed}.json';out=json.loads(dest.read_text()) if dest.exists() else dict(H=H,seed=seed,source_hashes=hashes,rows=[],coefficients={},attempts=0,cpu_seconds=0.)
 assert out['source_hashes']==hashes;prior=out['cpu_seconds'];out['attempts']+=1;out['status']='partial'
 def save():
  out['cpu_seconds']=prior+time.process_time()-start;out['hidden_forward_sequences']=212000*out['attempts'];dest.write_text(json.dumps(out,indent=2)+'\n')
 for row in q['rows']:
  family,K=row['family'],row['budget'];key=f'{family}_{K}'
  if row['solver']['converged'] or key in out['coefficients']:continue
  if time.process_time()-start>45:save();print('BOUNDED',H,seed,len(out['rows']),flush=True);return
  phi,_=features(train,family);phi=phi[:,:K];beta=np.asarray(q['coefficients'][key]);t=time.process_time();fixed,solver=fit(phi,htrain,beta,row['solver']['threshold'],max_steps=80)
  solver['initial_mse']=float(np.mean((np.maximum(phi@beta,0)-htrain)**2));solver['global_mse_increase']=solver['train_mse']-solver['initial_mse']
  pt,_=features(test,family);prediction=np.maximum(pt[:,:K]@fixed,0);error=((prediction-htest)**2).sum(0);r2=np.where(variance>1e-9,1-error/np.maximum(variance,1e-9),(error<1e-6).astype(float));good=(prediction@output.T).argmax(1)==truth;rate=float(good.mean());previous=(np.maximum(pt[:,:K]@beta,0)@output.T).argmax(1)==truth;delta=good.astype(float)-previous.astype(float);se=delta.std(ddof=1)/np.sqrt(len(delta))
  out['rows'].append(dict(family=family,budget=K,previous_accuracy=row['replacement_accuracy'],replacement_accuracy=rate,accuracy_ci=wilson(int(good.sum()),len(test)),gap_recovered=(rate-.1)/(old['model_accuracy']-.1),r2=r2.tolist(),n_r2_at_least_0_8=int((r2>=.8).sum()),accuracy_change=float(delta.mean()),paired_change_ci=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],solver=solver,fit_score_cpu=time.process_time()-t));out['coefficients'][key]=fixed.tolist();save();print(H,seed,family,K,rate,solver['converged'],solver['original_gradient_inf'],flush=True)
 out['status']='complete';save()
 for p in(op,qp):assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--H',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();main(a.H,a.seed)
