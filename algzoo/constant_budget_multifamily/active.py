"""Per-neuron active least squares and guarded complete piecewise line search."""
import numpy as np

def line(phi,y,beta,direction):
 a=phi@beta;b=phi@direction;ld=np.longdouble
 nonzero=b!=0;roots=np.zeros(len(b));roots[nonzero]=-a[nonzero]/b[nonzero];ids=np.flatnonzero(nonzero&np.isfinite(roots)&(roots>0)&(roots<1));ids=ids[np.argsort(roots[ids],kind='stable')]
 active=(a>0)|((a==0)&(b>0));aa=a.astype(ld);bb=b.astype(ld);yy=y.astype(ld)
 A=np.sum(bb[active]**2);B=np.sum(bb[active]*(aa[active]-yy[active]));C=np.sum((aa[active]-yy[active])**2)+np.sum(yy[~active]**2)
 sign=np.where(b[ids]>0,1,-1).astype(ld);As=np.r_[A,A+np.cumsum(sign*bb[ids]**2)];Bs=np.r_[B,B+np.cumsum(sign*bb[ids]*(aa[ids]-yy[ids]))];Cs=np.r_[C,C+np.cumsum(sign*(aa[ids]**2-2*aa[ids]*yy[ids]))]
 edges=np.r_[0.,roots[ids],1.].astype(ld);lo,hi=edges[:-1],edges[1:];stationary=lo.copy();positive=As>0;stationary[positive]=np.clip(-Bs[positive]/As[positive],lo[positive],hi[positive])
 candidates=np.r_[lo,hi,stationary];av=np.tile(As,3);bv=np.tile(Bs,3);cv=np.tile(Cs,3);scores=av*candidates*candidates+2*bv*candidates+cv;best=np.argsort(scores,kind='stable')[:8];alphas=np.unique(np.r_[0.,1.,np.asarray(candidates[best],dtype=float)])
 old=float(np.sum((np.maximum(a,0)-y)**2));bestloss=old;chosen=0.;newbeta=beta.copy();rejections=0
 for alpha in alphas:
  trial=beta+alpha*direction;loss=float(np.sum((np.maximum(phi@trial,0)-y)**2))
  if loss>old:rejections+=1
  if loss<bestloss or (loss==bestloss and chosen==0 and alpha>0):bestloss=loss;chosen=float(alpha);newbeta=trial
 return newbeta,dict(alpha=chosen,loss=bestloss,old_loss=old,breakpoints=len(ids),candidates=len(candidates),direct_candidates=len(alphas),increase_rejections=rejections,polynomial_min=float(np.min(scores)))

def fit(phi,h,beta,threshold,max_steps=20):
 beta=beta.copy();N,H=h.shape;histories=[];calls=0
 for j in range(H):
  y=h[:,j];v=beta[:,j].copy();records=[];stop='step cap'
  for step in range(max_steps):
   pre=phi@v;error=np.maximum(pre,0)-y;gradient=2*phi.T@(error*(pre>0))/(N*H);gn=float(np.max(abs(gradient)));calls+=1
   if gn<=threshold:stop='original gradient criterion';break
   mask=pre>0
   if not np.any(mask):stop='empty active set';break
   target,_,rank,_=np.linalg.lstsq(phi[mask],y[mask],rcond=None);direction=target-v
   if not np.any(direction):stop='zero least-squares direction';break
   candidate,meta=line(phi,y,v,direction);meta.update(step=step,gradient_before=gn,active=int(mask.sum()),active_rank=int(rank));records.append(meta)
   if meta['alpha']==0:stop='guarded line stalls';break
   v=candidate
  pre=phi@v;gradient=2*phi.T@((np.maximum(pre,0)-y)*(pre>0))/(N*H);gn=float(np.max(abs(gradient)));calls+=1;beta[:,j]=v
  histories.append(dict(neuron=j,steps=records,stop=stop,gradient_inf=gn,converged=gn<=threshold))
 pre=phi@beta;error=np.maximum(pre,0)-h;gradient=2*phi.T@(error*(pre>0))/(N*H)
 return beta,dict(train_mse=float(np.mean(error*error)),original_gradient_inf=float(np.max(abs(gradient))),threshold=threshold,converged=bool(np.max(abs(gradient))<=threshold),neurons=histories,per_neuron_gradient_calls=calls)
