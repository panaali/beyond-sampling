"""Exact moving first state followed by one analytic Gaussian transition."""
from pathlib import Path
import sys,time
import numpy as np
from scipy.special import ndtr,ndtri
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from engine import batched_terminal,fixed_target
from polynomial import conditional_expectations


def crossing_lines(model,axes,sign,limit):
    wi,wh,_=model
    if np.any(wi==0):raise ValueError('bounded primitive requires nonzero Wi')
    slope=wh@(wi*(wi*sign>0))
    lines=[(0.,-limit),(0.,limit),(1.,0.)]
    for d in range(2):
        lines.extend((-slope[d]/wi[d],float(x/wi[d])) for x in axes[d])
    return np.array(lines)


def moving_knots(model,axes,limit,target_limit,stats):
    points=[-target_limit,0.,target_limit]
    for sign in (-1,1):
        lo,hi=(-target_limit,0.) if sign<0 else (0.,target_limit)
        lines=crossing_lines(model,axes,sign,limit)
        stats['moving_affine_bound_lines']+=len(lines)
        for i,(a,b) in enumerate(lines):
            for c,d in lines[i+1:]:
                stats['moving_bound_pairs_examined']+=1
                if a!=c:
                    root=(d-b)/(a-c)
                    stats['moving_bound_intersections']+=1
                    if lo<root<hi:points.append(float(root))
    stats['moving_knot_candidates']+=len(points)
    return np.unique(points)


def direct_label_zero(data,q,operator_cache,deadline):
    model=data['model'];wi,wh,wo=model;stats=data['stats'];limit=data['limit']
    if len(wo)!=4:raise ValueError('bounded implementation requires T4')
    u=float(ndtri(ndtr(-limit)+(1-2*ndtr(-limit))*q))
    stats['conditional_inverse_normal_calls']+=1
    values=batched_terminal(model,data['states'][3],u,q,0,limit,stats,data['terminal_cache'])
    key=(2,float(q))
    if key not in operator_cache:
        operator_cache[key]=data['piece_cache'][2].matrices(u,q,stats,deadline)
    else:stats['polynomial_operator_cache_hits']+=1
    matrices,error=operator_cache[key]
    low,high=(matrix@values for matrix in matrices)
    stats['polynomial_sparse_value_products']+=2*sum(m.nnz for m in matrices)
    values2=np.stack((low[:,0],.5*low[:,1]+.5*high[:,0]),axis=1)
    values2=values2.reshape(len(data['axes'][2][0]),len(data['axes'][2][1]),2)
    h1=np.maximum(wi*u,0.)[None,:]
    low,high,bound=conditional_expectations(model,h1,data['axes'][2],values2,u,q,limit,stats,deadline)
    result=float((2*low[0,1]+high[0,0])/3)
    stats['direct_first_reachable_rows']+=1
    return result,(1+bound)*error+bound


def stream(data,q,weights,deadline,direct=True):
    stats=data['stats'];model=data['model'];T=len(model[2]);contributions={j:0. for j in data['labels']};bound=0.
    for qq,weight in zip(q,weights):
        if time.process_time()>deadline:raise TimeoutError('direct-first CPU cap')
        cache={}
        for j in data['labels']:
            if direct and j==0:value,error=direct_label_zero(data,float(qq),cache,deadline)
            else:value,error=fixed_target(model,j,float(qq),data['axes'],data['states'],8,data['limit'],stats,deadline,'polynomial_piece_cached',cache,data['piece_cache'],data['terminal_cache'])
            contributions[j]+=float(weight)*value/T;bound+=float(weight)*error/T
        size=sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for matrices,_ in cache.values() for m in matrices)
        stats['peak_streaming_conditional_kernel_bytes']=max(stats['peak_streaming_conditional_kernel_bytes'],size)
        stats['streaming_target_nodes']+=1
    return sum(contributions.values()),contributions,bound
