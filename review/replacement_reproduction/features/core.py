"""Independent fixed T258 endpoint helpers. No fitting, network or RNG code."""
import numpy as np


def _array(x,shape,dtype,name):
    if not isinstance(x,np.ndarray) or x.dtype!=dtype or x.shape!=shape or not np.isfinite(x).all():
        raise ValueError(name)


def features(x,t,f,hinges):
    if not isinstance(x,np.ndarray) or x.ndim!=2 or x.shape[0]<1:raise ValueError('input matrix')
    n=len(x);_array(x,(n,10),np.dtype('float64'),'finite float64 input')
    if type(t)is not int or not 1<=t<=10 or type(f)is not int or not 0<=f<12:raise ValueError('prefix/family')
    _array(hinges,(n,11,4),np.dtype('float32'),'stored float32 hinges')
    order=np.argsort(-x[:,:t],axis=1,kind='quicksort')
    ranked=np.zeros((n,10),dtype=np.float64);ages=np.full((n,10),t,dtype=np.int64)
    for k in range(t):
        ranked[:,k]=x[np.arange(n),order[:,k]];ages[:,k]=t-1-order[:,k]
    rank_count={0:0,1:0,2:1,3:2,4:3,5:3,6:4,7:4,8:3,9:3,10:3,11:2}[f]
    cell=np.zeros(n,dtype=np.int64)
    for k in range(rank_count):cell=cell*(t+1)+ages[:,k]
    if f in (5,6,7,8,9,10,11):cell=4*cell+np.minimum(np.count_nonzero(x[:,:t]>0,axis=1),3)
    cols=[x[:,t-1]]
    nr=3 if f in (0,1,2,3) else 10 if f==7 else 5
    cols += [ranked[:,k] for k in range(nr)]
    if f==1:
        theta=2*np.pi/8.42
        for k in range(3):cols += [ranked[:,k]*np.cos(theta*ages[:,k]),ranked[:,k]*np.sin(theta*ages[:,k])]
    if f in (5,6,8,9,10,11):
        cols += [np.maximum(ranked[:,5:t],0).sum(axis=1),np.minimum(ranked[:,5:t],0).sum(axis=1)]
    if f>=8:
        lag={8:1,9:3,10:9,11:9}[f]
        for ell in range(1,min(lag,t-1)+1):
            cols += [hinges[:,t-ell,j].astype(np.float64) for j in range(4)]
    return cell,np.stack(cols,axis=1)


def select(scores):
    _array(scores,(12,16),np.dtype('float64'),'finite twelve-family validation scores')
    choices=[]
    for stop in (12,8):
        chosen=np.zeros(16,dtype=np.int64)
        for j in range(16):
            best=scores[0,j]
            for f in range(1,stop):
                if scores[f,j]>best:chosen[j]=f;best=scores[f,j]
        choices.append(chosen)
    return tuple(choices)


def assemble(predictions,choices,readout):
    if not isinstance(predictions,np.ndarray) or predictions.ndim!=3 or predictions.shape[1]<1:raise ValueError('prediction tensor')
    n=predictions.shape[1];_array(predictions,(12,n,16),np.dtype('float64'),'finite prediction tensor')
    _array(choices,(16,),np.dtype('int64'),'integer choices')
    if np.any((choices<0)|(choices>=12)):raise ValueError('choice range')
    _array(readout,(10,16),np.dtype('float64'),'finite readout')
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        stored=predictions.astype(np.float32)
    if not np.isfinite(stored).all():raise ValueError('float32 prediction overflow')
    chosen=np.empty((n,16),dtype=np.float32)
    for j,f in enumerate(choices):chosen[:,j]=stored[f,:,j]
    hidden=np.maximum(chosen.astype(np.float64),0)
    logits=hidden@readout.T
    if not np.isfinite(logits).all():raise ValueError('readout overflow')
    return {'chosen':chosen,'hidden':hidden,'logits':logits,'prediction':np.argmax(logits,axis=1)}
