"""Positive finite-interval Gaussian quadrature, with separate tail bound."""
import math,time,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad_vec
from rank_mass import cdf_difference,coefficients,pair_owen

LIMIT=8.
def phi(x):return np.exp(-.5*np.asarray(x)**2)/math.sqrt(2*math.pi)
def direct(T,d,outer,inner):
    start=time.process_time();x,w=np.polynomial.legendre.leggauss(outer);y,v=np.polynomial.legendre.leggauss(inner)
    nodes=(y+1)/2;weights=v/2;total=np.zeros(4)
    stats={'rule_generations':2,'rule_nodes':outer+inner,'outer_nodes':outer,'inner_nodes':0,'cdf_values':0,'exp_values':0,'inverse_cdf_values':0}
    minpair=math.inf
    for u,weight in zip((x+1)*LIMIT/2,w*LIMIT/2):
        F=float(ndtr(u));G=F-.5;Q=float(ndtr(-u));stats['cdf_values']+=2;stats['exp_values']+=1
        if d==0:pair=(G*G,2*G*Q)
        else:
            points=u*nodes
            g0=u*float(weights@(phi(points)*cdf_difference(u,d*points)))
            high=min(u/d,LIMIT);width=high-u
            if width:
                points=u+width*nodes
                g1=G*Q+width*float(weights@(phi(points)*cdf_difference(u,d*points)))
                count=2*inner
            else:g1=G*Q;count=inner
            stats['inner_nodes']+=count;stats['exp_values']+=count;stats['cdf_values']+=2*count
            pair=(g0,g1)
        if min(pair)<0:raise ArithmeticError(('negative pair',T,d,u,pair))
        minpair=min(minpair,*pair)
        total+=weight*float(phi(u))*coefficients(T,d,u,F,pair);stats['cdf_values']+=4 if d else 2
    vector=[total[0]]*(T-4)+[total[1],total[1],total[2],total[3]]
    event=.25*(.25-math.atan(d)/(2*math.pi))
    return {'T':T,'d':d,'outer':outer,'inner':inner,'classes':total.tolist(),'event_mass':event,'partition_error':float(sum(vector)-event),'tail_bound_per_class':2*float(ndtr(-LIMIT)),'partition_tail_bound':2*T*float(ndtr(-LIMIT)),'minimum_pair_mass':minpair,'stats':stats,'cpu_seconds':time.process_time()-start}

def reference(T,d):
    calls=0
    def fun(u):
        nonlocal calls
        calls+=1;F=float(ndtr(u));pair=pair_owen(u,F,d)
        if min(pair)<-2e-15:raise ArithmeticError(('negative analytic pair',u,pair))
        return float(phi(u))*coefficients(T,d,u,F,pair)
    value,error,info=quad_vec(fun,0,LIMIT,epsabs=1e-13,epsrel=1e-11,full_output=True)
    return {'classes':value.tolist(),'estimated_error':float(error),'success':bool(info.success),'nodes':calls,'roundoff_not_certified':True}

def main():
    start=time.process_time();model=json.loads(Path(__file__).with_name('pilot_results.json').read_text());actual=float(__import__('fractions').Fraction(model['certificate']['d']))
    rows=[]
    for T,d in [(T,d) for T in (4,6,10) for d in (0.,.25,.5,1.)]+[(10,actual)]:
        ref=reference(T,d)
        for outer,inner in ((64,32),(128,32),(64,64),(128,64)):
            row=direct(T,d,outer,inner);row['reference']=ref;row['max_reference_difference']=float(np.max(np.abs(np.asarray(row['classes'])-ref['classes'])));rows.append(row)
    out={'rows':rows,'cpu_seconds':time.process_time()-start,'scope':'partial event only; tail bound excludes quadrature and roundoff','random_model_inputs':0}
    Path(__file__).with_name('direct_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'max_partition_error':max(abs(r['partition_error']) for r in rows),'max_reference_difference':max(r['max_reference_difference'] for r in rows),'max_partition_error_128_64':max(abs(r['partition_error']) for r in rows if r['outer']==128 and r['inner']==64),'trained':rows[-4:]},indent=2))
if __name__=='__main__':main()
