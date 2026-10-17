"""Fixed small-grid comparison, complete pair-rank target events."""
from pathlib import Path
from collections import defaultdict
import importlib.util,sys,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ndtr,ndtri
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from terminal import PairTerminal
path=HERE.parent.parent/'rank_conditioned_transport/measure_cycle/driver.py'
spec=importlib.util.spec_from_file_location('pair_full_measure',path)
outer=importlib.util.module_from_spec(spec);spec.loader.exec_module(outer)
from engine import fixed_target
from polynomial import fallback_rule


def pair_knots(objects,limit,stats):
    knots=[-limit,0.,limit]
    for obj in objects:
        for left,right,low,high in obj.pieces[False]:
            knots.extend((left,right))
            for a,b in (low,high):
                knots.extend((a*left+b,a*right+b));stats['pair_vertex_target_events']+=2
                if a!=1:
                    x=-b/(a-1);stats['pair_diagonal_intersections']+=1
                    if left<x<right:knots.append(x)
    stats['pair_target_candidates']+=len(knots)
    return np.unique([v for v in knots if -limit<=v<=limit])


def setup(model,h=3,seconds=100.):
    start=time.process_time();data=outer.setup(model,h,seconds);stats=data['stats'];T=len(model[2])
    if h!=3 or T!=4:raise ValueError('frozen scope is h3,T4')
    data['pairs']={j:[PairTerminal(model,s,j,limit=data['limit']) for s in data['states'][2]] for j in data['labels'] if j<2}
    objects=[obj for os in data['pairs'].values() for obj in os]
    events=pair_knots(objects,data['target_limit'],stats);data['points']=np.unique(np.r_[data['points'],events])
    data['pair_query_budget']=len(objects)*6*(len(data['points'])-1)
    if len(data['points'])-1>2000 or data['pair_query_budget']>50000:raise ValueError('frozen topology/query cost gate exceeded: '+str((len(data['points'])-1,data['pair_query_budget'])))
    data['stats']['complete_union_panels']=len(data['points'])-1
    data['stats']['predicted_pair_queries_gl2_plus_gl4']=data['pair_query_budget']
    data['stats']['setup_cpu_seconds']=time.process_time()-start
    return data


def pair_label(data,q,label,operators,deadline):
    model=data['model'];T=len(model[2]);stats=data['stats'];limit=data['limit']
    u=float(ndtri(ndtr(-limit)+(1-2*ndtr(-limit))*q));stats['conditional_inverse_normal_calls']+=1
    vals=[];errors=[]
    for obj in data['pairs'][label]:
        if time.process_time()>deadline:raise TimeoutError('pair transport query CPU cap')
        value,info=obj.probabilities(u);vals.append(value);errors.append(info['quadrature_error_estimates'])
    values=np.asarray(vals);bound=float(np.max(errors))
    for t in reversed(range(T-2)):
        if t==label:
            interp=RegularGridInterpolator(data['axes'][t+1],values.reshape(*(len(a) for a in data['axes'][t+1]),2),bounds_error=True)
            coords=np.maximum(data['states'][t]@model[1].T+model[0]*u,0.)
            values=interp(coords);stats['interpolation_queries']+=len(coords);stats['interpolators_built']+=1
        else:
            key=(t,float(q))
            if key not in operators:operators[key]=data['piece_cache'][t].matrices(u,q,stats,deadline)
            else:stats['polynomial_operator_cache_hits']+=1
            matrices,error=operators[key];low,high=(m@values for m in matrices)
            stats['polynomial_sparse_value_products']+=2*sum(m.nnz for m in matrices)
            m=T-t-int(label>=t)
            values=np.stack((low[:,0],(m-1)/m*low[:,1]+high[:,0]/m),axis=1)
            bound=(1+error)*bound+error
        if not np.all(np.isfinite(values)) or np.min(values)<-1e-10 or np.max(values)>1+1e-10:raise ArithmeticError('pair backward probability guard')
    return float(values[0,1]),bound


def estimate(model,order=2,paired=True,seconds=100.):
    start=time.process_time();deadline=start+seconds;fallback_rule.cache_clear();fallback_rule()
    data=setup(model,3,seconds);stats=data['stats'];T=len(model[2])
    q,weights,out=outer.rule(data['points'],T,order,'q',data['limit'],data['target_limit'])
    contributions={j:0. for j in data['labels']};bound=0.
    for qq,weight in zip(q,weights):
        if time.process_time()>deadline:raise TimeoutError('full pair cycle CPU cap')
        operators={}
        for label in data['labels']:
            if label==T-2:value,error=outer.pen.cached_label(data,float(qq),operators,deadline)
            elif paired:value,error=pair_label(data,float(qq),label,operators,deadline)
            else:value,error=fixed_target(model,label,float(qq),data['axes'],data['states'],8,data['limit'],stats,deadline,'polynomial_piece_cached',operators,data['piece_cache'],data['terminal_cache'])
            contributions[label]+=float(weight)*value/T;bound+=float(weight)*error/T
        stats['streaming_target_nodes']+=1
    pair_stats=defaultdict(float)
    for os in data['pairs'].values():
        for obj in os:
            for k,v in obj.stats.items():pair_stats[k]+=v
    stats['fallback_gauss12_rules_constructed']=1;stats['target_legendre_rules_constructed']=1
    stats['target_density_cdf_values']+=2*out['retained_panels']+3;stats['target_beta_density_polynomial_nodes']+=len(q)
    out.update(paired=paired,h=3,order=order,contributions=contributions,local_quadrature_diagnostic=bound,pair_work=dict(pair_stats),penultimate_work=dict(data['penultimate'].stats),work=dict(stats),cpu_seconds=time.process_time()-start,total_error_certified=False,portable_flops_certified=False)
    return sum(contributions.values()),out
