"""Fixed T201 reconstruction helpers. No imports of historical experiment code."""
import numpy as np

RCOND = 1e-12
LEGAL_KEYS = np.array([(c*9+p1)*9+p2 for c in range(3) for p1 in range(9)
                       for p2 in range(9) if p1 != p2], dtype=np.int64)
SLOT = np.full(243,-1,dtype=np.int64)
SLOT[LEGAL_KEYS] = np.arange(216)

class DomainError(ValueError): pass

def inputs(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=10 or len(x)==0 or not np.isfinite(x).all():
        raise DomainError('finite_nonempty_N_by_10_required')
    return x

def states(x):
    x=inputs(x)
    order=np.argsort(-x,axis=1,kind='stable')
    ranked=np.take_along_axis(x,order,axis=1)
    if np.any(np.diff(ranked,axis=1)==0):
        raise DomainError('input_tie_preserve_and_stop_no_resampling')
    head=np.argsort(-x[:,:9],axis=1,kind='stable')
    p1,p2=head[:,0],head[:,1]; rows=np.arange(len(x))
    m1,m2=x[rows,p1],x[rows,p2]; last=x[:,9]
    case=np.where(last>m1,0,np.where(last>m2,1,2))
    key=(case*9+p1)*9+p2; slot=SLOT[key]
    assert np.all(slot>=0)
    target=np.where(case==0,p1,np.where(case==1,9,p2))
    assert np.array_equal(target,order[:,1])
    inv=np.empty_like(order)
    inv[rows[:,None],order]=np.arange(1,11)[None,:]
    return dict(p1=p1,p2=p2,m1=m1,m2=m2,x9=last,case=case,key=key,slot=slot,
                target=target,positive_bin=np.minimum((x>0).sum(axis=1),2),global_rank=inv)

def lookup_design(s): return np.column_stack([s['m1'],s['m2'],s['x9']])

def separable_design(s):
    out=np.zeros((len(s['p1']),18)); rows=np.arange(len(out))
    out[rows,s['p2']]=s['m2']
    out[rows,9+s['p1']]=s['m1']-s['m2']
    return out

def hidden(h,n):
    h=np.asarray(h,dtype=np.float64)
    if h.ndim!=2 or h.shape[0]!=n or h.shape[1]==0 or not np.isfinite(h).all():
        raise DomainError('finite_matching_hidden_matrix_required')
    return h

def fit_lookup(s,h):
    X=lookup_design(s); h=hidden(h,len(X)); H=h.shape[1]
    counts=np.bincount(s['slot'],minlength=216)
    if np.any(counts<3): raise DomainError('lookup_state_underfilled')
    coef=np.zeros((216,3,H)); singular=[]; ranks=[]
    for k in range(216):
        at=s['slot']==k
        b,_,rank,sv=np.linalg.lstsq(X[at],h[at],rcond=RCOND)
        if rank!=3: raise DomainError(f'lookup_rank_deficient:{k}')
        coef[k]=b; ranks.append(int(rank)); singular.append(sv.tolist())
    return coef,dict(counts=counts.tolist(),ranks=ranks,singular_values=singular,rcond=RCOND)

def predict_lookup(s,coef):
    return np.einsum('ni,nij->nj',lookup_design(s),coef[s['slot']])

def fit_separable(s,h):
    X=separable_design(s); h=hidden(h,len(X)); mask=s['case']==2
    if mask.sum()<18: raise DomainError('separable_underfilled')
    b,_,rank,sv=np.linalg.lstsq(X[mask],h[mask],rcond=RCOND)
    if rank!=18: raise DomainError('separable_rank_deficient')
    return b,dict(count=int(mask.sum()),rank=int(rank),singular_values=sv.tolist(),rcond=RCOND)

def forward_final(x,Wi,Wh):
    x=inputs(x); Wi=np.asarray(Wi,dtype=np.float64); Wh=np.asarray(Wh,dtype=np.float64)
    if Wi.ndim!=1 or Wh.shape!=(len(Wi),len(Wi)) or not np.isfinite(Wi).all() or not np.isfinite(Wh).all():
        raise DomainError('invalid_weights')
    h=np.zeros((len(x),len(Wi)))
    for t in range(10): h=np.maximum(h@Wh.T+x[:,t,None]*Wi[None,:],0)
    if not np.isfinite(h).all(): raise DomainError('nonfinite_hidden')
    return h

def hidden_metrics(h,estimate):
    h=np.asarray(h,dtype=np.float64); estimate=hidden(estimate,len(h));h=hidden(h,len(h))
    if h.shape!=estimate.shape or len(h)==0: raise DomainError('metric_shape_or_empty')
    sse=((estimate-h)**2).sum(axis=0); sst=((h-h.mean(axis=0))**2).sum(axis=0)
    r2=[None if d==0 else float(1-n/d) for n,d in zip(sse,sst,strict=True)]
    valid=[v for v in r2 if v is not None]
    return dict(n=len(h),sse=sse.tolist(),sst=sst.tolist(),mse=(sse/len(h)).tolist(),
                r2=r2,median_r2=float(np.median(valid)) if valid else None,
                zero_variance_columns=[j for j,d in enumerate(sst) if d==0])

def cross_tab(s,pred):
    pred=np.asarray(pred)
    if pred.shape!=s['target'].shape or pred.dtype.kind not in 'iu' or np.any((pred<0)|(pred>=10)):
        raise DomainError('integer_predictions_0_to_9_required')
    ranks=s['global_rank'][np.arange(len(pred)),pred]
    table=np.zeros((3,3,10),dtype=np.int64)
    np.add.at(table,(s['positive_bin'],s['case'],ranks-1),1)
    correct=int((pred==s['target']).sum())
    assert table.sum()==len(pred) and table[:,:,1].sum()==correct
    return dict(n=len(pred),correct=correct,table=table.tolist())
