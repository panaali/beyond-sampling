"""Sufficient first-label predicates for finite bias-free, zero-initial ReLU RNNs."""
from collections import Counter
import numpy as np

def recognize(model):
 counts=Counter()
 def check(value):counts['metadata_predicate']+=1;return bool(value)
 def result(status,**kw):return dict(status=status,counts=dict(counts),**kw)
 if not check(isinstance(model,(tuple,list))):return result('rejected',reason='Expected three-array model')
 if not check(len(model)==3):return result('rejected',reason='Expected Wi,Wh,Wo')
 for part in model:
  if not check(isinstance(part,np.ndarray)):return result('rejected',reason='NumPy arrays required')
  if not check(part.dtype==np.dtype('float64')):return result('rejected',reason='Stored binary64 weights required')
 Wi,Wh,Wo=model
 for part,ndim in ((Wi,1),(Wh,2),(Wo,2)):
  if not check(part.ndim==ndim):return result('rejected',reason='Weight rank mismatch')
 H=Wi.shape[0];T=Wo.shape[0]
 if not check(H>0):return result('rejected',reason='Positive hidden width required')
 if not check(T>=2):return result('rejected',reason='At least two input positions required')
 if not check(Wh.shape==(H,H)):return result('rejected',reason='Recurrent weight shape mismatch')
 if not check(Wo.shape[1]==H):return result('rejected',reason='Readout width mismatch')
 flags=[np.isfinite(part).all() for part in model];size=sum(part.size for part in model);counts['finite_checks']+=size;counts['boolean_reduction']+=size-1
 if not bool(flags[0]&flags[1]&flags[2]):return result('rejected',reason='Nonfinite weight')
 zero=Wi==0.;counts['scalar_comparison']+=H;counts['boolean_reduction']+=H-1
 if bool(zero.all()):reason='zero_input_weights'
 else:
  dominance=Wo[0][None,:]>=Wo[1:];n=(T-1)*H;counts['scalar_comparison']+=n;counts['boolean_reduction']+=n-1
  if not bool(dominance.all()):return result('fallback',reason='Sufficient constant-first-label predicates do not hold')
  reason='row0_coordinatewise_dominance_including_identical_rows'
 counts['scalar_division']+=1
 return result('certified',reason=reason,prediction=0,accuracy=1./T,exact_accuracy=[1,T],contract='Bias-free ReLU recurrence with zero initial hidden state; iid continuous inputs, or exchangeable inputs with almost-surely distinct coordinates; first-index argmax; real-arithmetic model semantics')

def dispatch(model,fallback):
 certificate=recognize(model)
 if certificate['status']=='certified':return dict(shortcut=certificate,result=certificate['accuracy'],fallback_called=False)
 if certificate['status']=='rejected':return dict(shortcut=certificate,result=None,fallback_called=False)
 return dict(shortcut=certificate,result=fallback(model),fallback_called=True)
