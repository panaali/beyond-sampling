"""Independent finite binary64 recurrence and descriptive paired counts."""
import numpy as np

def forward(x,wi,wh,wo):
    if x.dtype!=np.float64 or x.ndim!=2 or x.shape[1]!=10 or not np.isfinite(x).all():raise ValueError('finite float64 ten-input rows')
    n=len(x);h=np.zeros((n,16));pre=np.zeros((n,11,16));hidden=np.zeros_like(pre)
    for j in range(10):
        recurrent=h.dot(wh.T);forced=np.multiply(x[:,j,None],wi[None,:]);z=np.add(recurrent,forced)
        pre[:,j+1]=z;h=np.maximum(z,0.);hidden[:,j+1]=h
    logits=h.dot(wo.T);labels=logits.argmax(axis=1)
    if not np.isfinite(pre).all() or not np.isfinite(logits).all():raise ValueError('nonfinite recurrence')
    return {'pre':pre,'hidden':hidden,'logits':logits,'labels':labels,'gates':np.packbits((pre[:,1:]>0).reshape(n,160),axis=1,bitorder='little')}

def scores(target,pred):
    sst=((target-target.mean(axis=0))**2).sum(axis=0);sse=((pred-target)**2).sum(axis=0)
    return 1-sse/np.maximum(sst,1e-300),sse,sst

def metrics(x,original,pred):
    target=np.argsort(x,axis=1)[:,-2];a=original==target;b=pred==target;same=original==pred
    table=[int(np.count_nonzero(a&b)),int(np.count_nonzero(a&~b)),int(np.count_nonzero(~a&b)),int(np.count_nonzero(~a&~b&same)),int(np.count_nonzero(~a&~b&~same))]
    assert sum(table)==len(x)
    order=np.sort(x,axis=1);near=(order[:,-1]-order[:,-2]<.05)|(order[:,-2]-order[:,-3]<.05)|((x>0).sum(axis=1)<=1)
    return {'N':len(x),'original_correct':int(a.sum()),'replacement_correct':int(b.sum()),'agreement':int(same.sum()),'disjoint_outcomes':table,'near_union_count':int(near.sum()),'near_disagreement':int(np.count_nonzero(near&~same)),'input_tie_rows':int(np.count_nonzero(np.any(order[:,1:]==order[:,:-1],axis=1)))}
