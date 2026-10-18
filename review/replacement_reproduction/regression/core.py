"""Independent grouped normal equations and deployable cell regression codec."""
import json
import numpy as np

def data(c,v,y=None):
    if not isinstance(c,np.ndarray) or c.dtype!=np.int64 or c.ndim!=1 or not len(c):raise ValueError('nonempty int64 cells')
    if not isinstance(v,np.ndarray) or v.dtype!=np.float64 or v.ndim!=2 or v.shape[0]!=len(c) or not 1<=v.shape[1]<=64 or not np.isfinite(v).all():raise ValueError('finite float64 design')
    if y is not None and (not isinstance(y,np.ndarray) or y.dtype!=np.float64 or y.ndim!=2 or y.shape[0]!=len(c) or not 1<=y.shape[1]<=16 or not np.isfinite(y).all()):raise ValueError('finite float64 targets')

def fit(c,v,y,emit):
    data(c,v,y);ids=np.unique(c);d=v.shape[1]
    counts=[];grams=[];cross=[]
    for label in ids:
        keep=c==label;x=v[keep];target=y[keep]
        counts.append(int(keep.sum()));grams.append(x.T@x);cross.append(x.T@target)
    grams=np.stack(grams);cross=np.stack(cross);G=grams.sum(axis=0);B=cross.sum(axis=0)
    systems=[]
    def solve(label,a,b):
        systems.append({'label':label,'matrix':a.copy(),'rhs':b.copy()});emit('solve_started',label,a,b)
        z=np.linalg.solve(a,b);emit('solve_completed',label,z,None)
        if not np.isfinite(z).all():raise ValueError('nonfinite solve')
        return z
    pooled_matrix=G+(1e-9*np.trace(G)/d)*np.eye(d)
    pooled=solve('pooled',pooled_matrix,B)
    coef=[]
    for k,label in enumerate(ids):
        a=grams[k]+(1e-6*(np.trace(grams[k])/d+1e-12))*np.eye(d)
        coef.append(solve('cell:'+str(int(label)),a,cross[k]))
    return {'schema':1,'ids':ids,'counts':np.array(counts,dtype=np.int64),'grams':grams,'cross':cross,'pooled':pooled,'coef':np.stack(coef),'systems':systems,'d':d,'q':y.shape[1]}

def predict(model,c,v):
    data(c,v)
    if v.shape[1]!=model['d']:raise ValueError('prediction dimension')
    ans=v@model['pooled'];seen=np.zeros(len(c),dtype=bool)
    for i,label in enumerate(model['ids']):
        keep=c==label;ans[keep]=v[keep]@model['coef'][i];seen|=keep
    return ans,seen

def dumps(model):
    payload={k:model[k] for k in ['schema','d','q']}
    payload.update({k:model[k].tolist() for k in ['ids','counts','pooled','coef']})
    return json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)

def loads(text):
    z=json.loads(text)
    if not isinstance(z,dict) or set(z)!={'schema','d','q','ids','counts','pooled','coef'} or type(z['schema']) is not int or z['schema']!=1:raise ValueError('codec schema')
    if type(z['d']) is not int or not 1<=z['d']<=64 or type(z['q']) is not int or not 1<=z['q']<=16:raise ValueError('codec dimensions')
    ids=z['ids'];counts=z['counts']
    if not isinstance(ids,list) or not ids or any(type(x)is not int or not -2**63<=x<2**63 for x in ids) or any(a>=b for a,b in zip(ids,ids[1:])):raise ValueError('codec ids')
    if not isinstance(counts,list) or len(counts)!=len(ids) or any(type(x)is not int or not 0<x<2**63 for x in counts):raise ValueError('codec counts')
    pooled=np.asarray(z['pooled'],dtype=np.float64);coef=np.asarray(z['coef'],dtype=np.float64)
    if pooled.shape!=(z['d'],z['q']) or coef.shape!=(len(ids),z['d'],z['q']) or not np.isfinite(pooled).all() or not np.isfinite(coef).all():raise ValueError('codec coefficients')
    return {'schema':1,'d':z['d'],'q':z['q'],'ids':np.array(ids,dtype=np.int64),'counts':np.array(counts,dtype=np.int64),'pooled':pooled,'coef':coef}
