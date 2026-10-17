from pathlib import Path
from fractions import Fraction as F
import json,time
import numpy as np
from rank_mass import joint_mass,pair_positive,pair_owen

def add(a,b):
    out=[F(0)]*max(len(a),len(b))
    for i,x in enumerate(a):out[i]+=x
    for i,x in enumerate(b):out[i]+=x
    return out
def mul(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out
def coefficient(ordinary,positive):
    zero=[F(1)];one=[F(0)]
    for a in [[F(0),F(1)]]*ordinary+[[F(-1,2),F(1)]]*positive:
        b=[F(1),F(-1)];one=add(mul(one,a),mul(zero,b));zero=mul(zero,a)
    return sum((v/F(i+1)*(1-F(1,2)**(i+1)) for i,v in enumerate(one)),F(0))
def d0_reference(T):
    m=T-4;A=coefficient(m-1,4) if m else F(0);B=coefficient(m,3)
    if m*A+4*B!=F(1,16):raise AssertionError('exact sign partition')
    return [float(A),float(B),float(B),float(B)]

def main():
    start=time.process_time();rows=[]
    for T in (4,6,10):
        for d in (0.,.25,.5,1.):
            for outer,inner in ((32,32),(64,32),(32,64),(64,64)):
                r=joint_mass(T,d,outer,inner)
                if min(r['joint_by_target'])<0:raise AssertionError('negative target mass')
                if d==0:
                    r['exact_d0_classes']=d0_reference(T);r['max_d0_error']=max(abs(a-b) for a,b in zip(r['classes'],r['exact_d0_classes']))
                    if r['max_d0_error']>1e-14:raise AssertionError('polynomial control fails')
                rows.append(r)
    pairrows=[];x,w=np.polynomial.legendre.leggauss(64)
    from scipy.special import ndtr
    for d in (.25,.5,1.):
        for u in (.01,.3,1.,3.):
            stats={'inverse_cdf_values':0,'cdf_values':0,'inner_nodes':0};positive=pair_positive(u,float(ndtr(u)),d,(x+1)/2,w/2,stats);closed=pair_owen(u,float(ndtr(u)),d)
            pairrows.append({'d':d,'u':u,'positive':positive,'owen':closed,'max_difference':max(abs(a-b) for a,b in zip(positive,closed)),'stats':stats})
    result={'cases':rows,'pair_checks':pairrows,'cpu_seconds':time.process_time()-start,'sampled_inputs':0}
    Path(__file__).with_name('controls_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':result['cpu_seconds'],'max_partition_error':max(abs(r['partition_error']) for r in rows),'max_d0_error':max(r.get('max_d0_error',0) for r in rows),'max_pair_difference':max(r['max_difference'] for r in pairrows)},indent=2))
if __name__=='__main__':main()
