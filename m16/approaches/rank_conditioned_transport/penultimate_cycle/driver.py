"""Private q-panel wrapper; earlier recurrence unchanged for each label."""
import importlib.util
from pathlib import Path
import sys,time
import numpy as np
from scipy.special import ndtr,ndtri
from cache import PenultimateCache
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from breakpoints import prepare
from engine import fixed_target
from polynomial import fallback_rule


def cached_label(data,q,operators,deadline):
    model=data['model'];T=len(model[2]);stats=data['stats'];limit=data['limit']
    u=float(ndtri(ndtr(-limit)+(1-2*ndtr(-limit))*q))
    stats['conditional_inverse_normal_calls']+=1
    values=data['penultimate'].values(u,q);bound=0.
    for t in reversed(range(T-2)):
        key=(t,float(q))
        if key not in operators:operators[key]=data['piece_cache'][t].matrices(u,q,stats,deadline)
        else:stats['polynomial_operator_cache_hits']+=1
        matrices,error=operators[key];low,high=(m@values for m in matrices)
        stats['polynomial_sparse_value_products']+=2*sum(m.nnz for m in matrices)
        m=T-t-1
        values=np.stack((low[:,0],(m-1)/m*low[:,1]+high[:,0]/m),axis=1)
        bound=(1+error)*bound+error
    return float(values[0,1]),bound


def estimate(model,h=9,order=2,cached=True,seconds=100.):
    start=time.process_time();memo={}
    sys.path.insert(0,str(HERE.parent/'source_mesh_cycle'))
    try:
        spec=importlib.util.spec_from_file_location('penultimate_private_outer',HERE.parent/'source_mesh_cycle/driver.py')
        outer=importlib.util.module_from_spec(spec);spec.loader.exec_module(outer)
    finally:sys.path.pop(0)
    def setup(*args,**kwargs):
        data=prepare(*args,**kwargs)
        cache=PenultimateCache(model,data['states'][len(model[2])-2],data['limit'],data['target_limit'],seconds-(time.process_time()-start))
        data['penultimate']=cache;memo['cache']=cache
        data['stats']['fixed_topology_before_penultimate']=len(data['points'])-1
        data['points']=np.unique(np.r_[data['points'],cache.knots])
        return data
    def stream(data,q,weights,deadline):
        stats=data['stats'];T=len(model[2]);contributions={j:0. for j in data['labels']};total_bound=0.
        for qq,weight in zip(q,weights):
            if time.process_time()>deadline:raise TimeoutError('penultimate CPU cap')
            operators={}
            for label in data['labels']:
                if cached and label==T-2:value,bound=cached_label(data,float(qq),operators,deadline)
                else:value,bound=fixed_target(model,label,float(qq),data['axes'],data['states'],8,data['limit'],stats,deadline,'polynomial_piece_cached',operators,data['piece_cache'],data['terminal_cache'])
                contributions[label]+=float(weight)*value/T;total_bound+=float(weight)*bound/T
            size=sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for matrices,_ in operators.values() for m in matrices)
            stats['peak_streaming_conditional_kernel_bytes']=max(stats['peak_streaming_conditional_kernel_bytes'],size)
            stats['streaming_target_nodes']+=1
        return sum(contributions.values()),contributions,total_bound
    outer.prepare=setup;outer.stream=stream
    fallback_rule.cache_clear();fallback_rule()
    value,stats=outer.estimate(model,h,order,'baseline',seconds=seconds-(time.process_time()-start))
    stats['penultimate_cached']=cached;stats['penultimate_work']=dict(memo['cache'].stats)
    stats['work']['fallback_gauss12_rules_constructed']=1;stats['cpu_seconds']=time.process_time()-start
    return value,stats
