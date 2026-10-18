"""Minimum-norm active correction; original active.py remains unchanged."""
import numpy as np
from active import line

def fit(phi,h,beta,threshold,max_steps=80):
 beta=beta.copy();N,H=h.shape;histories=[];calls=0
 for j in range(H):
  y=h[:,j];v=beta[:,j].copy();records=[];stop='step cap'
  for step in range(max_steps):
   pre=phi@v;error=np.maximum(pre,0)-y;gradient=2*phi.T@(error*(pre>0))/(N*H);gn=float(np.max(abs(gradient)));calls+=1
   if gn<=threshold:stop='original gradient criterion';break
   mask=pre>0
   if not np.any(mask):stop='empty active set';break
   direction,_,rank,singular=np.linalg.lstsq(phi[mask],y[mask]-pre[mask],rcond=None);cutoff=float(np.finfo(float).eps*max(phi[mask].shape)*singular[0])
   if not np.any(direction):stop='zero least-squares direction';break
   candidate,meta=line(phi,y,v,direction);meta.update(step=step,gradient_before=gn,active=int(mask.sum()),active_rank=int(rank),numerical_rank_cutoff=cutoff,smallest_singular_value=float(singular[-1]));records.append(meta)
   if meta['alpha']==0:stop='guarded line stalls';break
   v=candidate
  pre=phi@v;gradient=2*phi.T@((np.maximum(pre,0)-y)*(pre>0))/(N*H);gn=float(np.max(abs(gradient)));calls+=1;beta[:,j]=v
  histories.append(dict(neuron=j,steps=records,stop=stop,gradient_inf=gn,converged=gn<=threshold))
 pre=phi@beta;error=np.maximum(pre,0)-h;gradient=2*phi.T@(error*(pre>0))/(N*H)
 return beta,dict(train_mse=float(np.mean(error*error)),original_gradient_inf=float(np.max(abs(gradient))),threshold=threshold,converged=bool(np.max(abs(gradient))<=threshold),neurons=histories,per_neuron_gradient_calls=calls)
