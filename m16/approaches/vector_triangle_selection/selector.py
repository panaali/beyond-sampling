"""Scalar radial metadata; vector trials with original scalar decision guards."""
import math,time
import numpy as np
from radial_probe import Grouped,scalar_radial

ORDERS=(4,8,12,16,24,32,48,64,96,128)


def scalar_bound(mid,ratio,n):
    next_ratio=ratio*(n+3)/(n+1)
    if next_ratio>=1:return math.inf
    return 0. if ratio==0 else ((n+1)*(n+2)/2)*(ratio**n)/(mid*mid*mid*(1-next_ratio))


class Guarded(Grouped):
    def __init__(self,audit,max_nodes=65536):
        super().__init__(audit,max_nodes);self.scalar_rescues=0;self.scalar_checks=0;self.vector_trials=0;self._queue=None

    def select(self,vertices,exponent,tolerance,max_order):
        if self._queue is not None:return next(self._queue)
        return super().select(vertices,exponent,tolerance,max_order)

    def select_many(self,vertices,exponent,tolerance,max_order):
        if exponent!=6 or vertices.ndim!=3 or vertices.shape[1:]!=(3,6):raise ValueError('dimension6 triangles required')
        if not math.isfinite(tolerance) or tolerance<0:raise ValueError('finite nonnegative tolerance required')
        if not len(vertices):return []
        radial=np.array([scalar_radial(v) for v in vertices]);mid=radial[:,2];ratio=radial[:,3]
        ns=np.array([n for n in ORDERS if n<=max_order],dtype=int)
        if not len(ns):raise ArithmeticError('positive triangle error bound unavailable')
        with np.errstate(divide='ignore',invalid='ignore',over='ignore',under='ignore'):
            next_ratio=ratio[:,None]*(ns+3)/(ns+1)
            errors=((ns+1)*(ns+2)/2)*ratio[:,None]**ns/(mid[:,None]*mid[:,None]*mid[:,None]*(1-next_ratio))
        valid=(next_ratio<1)&np.isfinite(errors)&(errors>=0)
        passed=valid&(errors<=tolerance);any_pass=np.any(passed,axis=1);chosen=np.argmax(passed,axis=1)
        self.vector_trials+=len(vertices)*len(ns)
        self.audit.add('vector_bound_trials_and_guards',180*len(vertices)*len(ns))
        self.audit.add('vector_selection_setup',64*len(vertices)+64)
        metas=[];eps=np.finfo(float).eps
        for i,v in enumerate(vertices):
            lo,hi,mm,rr=map(float,radial[i]);index=int(chosen[i]);n=int(ns[index]);rescue=not any_pass[i]
            rescue|=not (math.isfinite(mm) and mm>0 and 0<=rr<1 and lo>0 and hi>=lo)
            trial_next=next_ratio[i,:index+1]
            rescue|=bool(np.any(abs(trial_next-1)<=512*eps))
            trial_errors=errors[i,:index+1]
            rescue|=bool(np.any(abs(trial_errors-tolerance)<=512*eps*np.maximum(abs(trial_errors),abs(tolerance))))
            if not rescue:
                error=scalar_bound(mm,rr,n);previous=math.inf if index==0 else scalar_bound(mm,rr,int(ns[index-1]))
                self.scalar_checks+=1+int(index>0);self.audit.add('scalar_selected_previous_checks',120*(1+int(index>0)))
                rescue=not (math.isfinite(error) and error<=tolerance and previous>tolerance)
            if rescue:
                self.scalar_rescues+=1;self.audit.add('scalar_rescue_duplicate_radial_setup',512)
                meta=super().select(v,exponent,tolerance,max_order)
            else:
                self.audit.events['positive_fallback_calls']+=1
                self.audit.add('positive_radial_setup',512);self.audit.add('positive_bound_trials',120*(index+1))
                meta=dict(quadrature_nodes=n*n,quadrature_error_bound=error,quadrature_order=n,
                          legendre_rule_generations=0,legendre_rule_generation_nodes=0,
                          qmin=lo,qmax=hi,binomial_ratio=rr,bound_trials=index+1,
                          bound_lgamma_calls=0,bound_exp_calls=0,bound_log_calls=0)
            metas.append(meta)
        return metas

    def many(self,vertices,exponent,tolerance,max_order):
        if self._queue is not None:raise RuntimeError('selection queue is not reentrant')
        vertices=np.asarray(vertices);start=time.process_time()
        metas=self.select_many(vertices,exponent,tolerance,max_order)
        self.selection_cpu+=time.process_time()-start
        self._queue=iter(metas)
        try:return super().many(vertices,exponent,tolerance,max_order)
        finally:self._queue=None
