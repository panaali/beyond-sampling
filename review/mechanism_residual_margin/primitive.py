"""T201 saved-array diagnostic, never a forward or estimator."""
import numpy as np

TRANSITIONS = ('both_correct','network_error_erased','error_introduced',
               'both_wrong_same_class','both_wrong_different_class')

class DomainError(ValueError): pass

def analyze(h, hh, logits, lookup_logits, wo, truth):
    h,hh,logits,lookup_logits,wo=[np.asarray(a,dtype=np.float64) for a in (h,hh,logits,lookup_logits,wo)]
    truth=np.asarray(truth)
    if h.ndim!=2 or len(h)==0 or h.shape[1]==0 or hh.shape!=h.shape:
        raise DomainError('nonempty matching hidden matrices required')
    n,H=h.shape
    if wo.ndim!=2 or wo.shape[1]!=H or wo.shape[0]<2:
        raise DomainError('readout shape')
    C=wo.shape[0]
    if logits.shape!=(n,C) or lookup_logits.shape!=(n,C):raise DomainError('logit shape')
    if truth.shape!=(n,) or truth.dtype.kind not in 'iu' or np.any(truth<0) or np.any(truth>=C):
        raise DomainError('integer truth indices required')
    if not all(np.isfinite(a).all() for a in (h,hh,logits,lookup_logits,wo)):
        raise DomainError('finite arrays required')
    rows=np.arange(n)
    pred=np.argmax(logits,axis=1); look=np.argmax(lookup_logits,axis=1)
    competing=logits.copy();competing[rows,truth]=-np.inf
    competitor=np.argmax(competing,axis=1)
    nc=pred==truth;lc=look==truth
    transition=np.where(nc,np.where(lc,0,2),np.where(lc,1,np.where(pred==look,3,4)))
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        residual=hh-h
        direction=wo[truth]-wo[competitor]
        terms=residual*direction
        total=terms[:,0].copy();absolute=np.abs(terms[:,0])
        for j in range(1,H):
            total=total+terms[:,j]
            absolute=absolute+np.abs(terms[:,j])
        original_margin=logits[rows,truth]-logits[rows,competitor]
        lookup_margin=lookup_logits[rows,truth]-lookup_logits[rows,competitor]
        delta=lookup_margin-original_margin
        closure=delta-total
    if not all(np.isfinite(a).all() for a in (terms,total,absolute,original_margin,lookup_margin,delta,closure)):
        raise DomainError('nonfinite derived arithmetic')
    return dict(network=pred,lookup=look,competitor=competitor,transition=transition,
                terms=terms,term_sum=total,term_abs_sum=absolute,
                original_margin=original_margin,lookup_margin=lookup_margin,
                margin_delta=delta,closure_residual=closure,
                network_max_ties=(logits==logits[rows,pred,None]).sum(axis=1),
                lookup_max_ties=(lookup_logits==lookup_logits[rows,look,None]).sum(axis=1),
                named_arithmetic=dict(subtractions=2*n*H+4*n,multiplications=n*H,
                    additions=2*n*(H-1),absolute_values=n*H),
                count_scope='Named residual/margin arithmetic only; validation, argmax, masking, array allocation, indices, predicates, reductions for tie counts and runtime excluded.')
